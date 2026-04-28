"""
Step 2+5: Control Loop Latency Experiment
==========================================
Runs a fixed-rate loop at 100 Hz and measures:
  1. Loop jitter     — how consistent the timing actually is over USB serial
  2. Round-trip lag  — time from Goal_Position write until Present_Position
                       starts moving (servo firmware + serial latency combined)

What to observe:
  - Jitter plot: are loop intervals clustered tightly around 10 ms?
  - Lag: how many milliseconds before the servo starts responding?
  - The lag you see here is the dead time your outer PD loop must fight.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

# ── Config ────────────────────────────────────────────────────────────────────
PORT = "/dev/ttyACM0"
LOOP_HZ = 100
STEP_DEG = 30          # small step so the servo doesn't swing far
SETTLE_S = 3.0         # how long to record after the step
MOVE_THRESHOLD_DEG = 0.5  # position change that counts as "started moving"

TICKS_PER_DEG = 4096 / 360
# ──────────────────────────────────────────────────────────────────────────────

bus = FeetechMotorsBus(
    port=PORT,
    motors={"shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
)
bus.connect()

start_ticks = bus.read("Present_Position", "shoulder_pan", normalize=False)
target_ticks = int(start_ticks + STEP_DEG * TICKS_PER_DEG)
print(f"Start: {start_ticks}  Target: {target_ticks}  (+{STEP_DEG}°)")
input("Clear space for a small 10° move, then press ENTER.")

times = []          # wall time of each loop tick
intervals = []      # actual time between ticks (reveals jitter)
positions_deg = []  # position relative to start

step_sent_t = None  # when we sent the goal command
lag_s = None        # measured round-trip lag

dt = 1.0 / LOOP_HZ
step_sent = False
t_prev = time.perf_counter()
t_start = t_prev

# ── Fixed-rate loop ───────────────────────────────────────────────────────────
while True:
    t_now = time.perf_counter()
    elapsed = t_now - t_start

    # Send the step command on the second tick (gives one clean baseline read)
    if not step_sent and elapsed > dt:
        step_sent_t = time.perf_counter()
        bus.write("Goal_Position", "shoulder_pan", target_ticks, normalize=False)
        step_sent = True

    raw = bus.read("Present_Position", "shoulder_pan", normalize=False)
    pos_deg = (raw - start_ticks) / TICKS_PER_DEG

    times.append(elapsed)
    intervals.append(t_now - t_prev)
    positions_deg.append(pos_deg)

    # Detect first movement to measure lag
    if step_sent and lag_s is None and abs(pos_deg) > MOVE_THRESHOLD_DEG:
        lag_s = time.perf_counter() - step_sent_t

    t_prev = t_now

    if step_sent and elapsed > SETTLE_S:
        break

    # Sleep for the remainder of this tick
    next_tick = t_start + len(times) * dt
    sleep_s = next_tick - time.perf_counter()
    if sleep_s > 0:
        time.sleep(sleep_s)

bus.disconnect()

# ── Results ───────────────────────────────────────────────────────────────────
intervals_ms = np.array(intervals[1:]) * 1000  # skip first (no prev)
print(f"\nLoop timing over {len(intervals_ms)} ticks:")
print(f"  mean interval : {intervals_ms.mean():.2f} ms  (target {1000/LOOP_HZ:.1f} ms)")
print(f"  std  (jitter) : {intervals_ms.std():.2f} ms")
print(f"  max            : {intervals_ms.max():.2f} ms")

if lag_s is not None:
    print(f"\nRound-trip lag : {lag_s*1000:.1f} ms")
else:
    print("\nRound-trip lag : not detected (servo may not have moved)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

# Top: position over time
ax1.axhline(STEP_DEG, color="r", linestyle="--", linewidth=1, label=f"target ({STEP_DEG}°)")
if lag_s is not None:
    ax1.axvline(lag_s, color="orange", linestyle=":", linewidth=1.5, label=f"first movement ({lag_s*1000:.0f} ms)")
ax1.plot(times, positions_deg, linewidth=1.2, label="measured position")
ax1.set_ylabel("position (°)")
ax1.set_title("Round-trip Latency — shoulder_pan")
ax1.legend()
ax1.grid(True, alpha=0.4)

# Bottom: loop interval histogram
ax2.hist(intervals_ms, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
ax2.axvline(1000 / LOOP_HZ, color="r", linestyle="--", label=f"target ({1000/LOOP_HZ:.0f} ms)")
ax2.set_xlabel("loop interval (ms)")
ax2.set_ylabel("count")
ax2.set_title(f"Loop Jitter  —  mean={intervals_ms.mean():.2f} ms  std={intervals_ms.std():.2f} ms")
ax2.legend()
ax2.grid(True, alpha=0.4)

fig.tight_layout()
fig.savefig("latency.png", dpi=150)
print("Saved latency.png")
