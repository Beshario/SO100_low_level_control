"""
Step 1: Step Response Experiment
=================================
Commands shoulder_pan (ID 1) from its current position +30°,
logs (time, position) at ~100 Hz, and saves a plot.

What to observe:
  - Rise time: how long to reach 30°?
  - Overshoot: does it go past 30° before settling?
  - Oscillation: does it ring, or settle cleanly?
  - Delay: is there a dead zone at the start before it begins moving?
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

# ── Config ────────────────────────────────────────────────────────────────────
PORT = "/dev/ttyACM0"   # change to match your port (check: ls /dev/tty{USB,ACM}*)
STEP_DEG = 180           # size of the commanded step in degrees
LOG_DURATION_S = 4.0    # how long to record after the step command

# STS3215 resolution: 4096 ticks per full revolution
TICKS_PER_DEG = 4096 / 360
# ──────────────────────────────────────────────────────────────────────────────

bus = FeetechMotorsBus(
    port=PORT,
    # Only connecting shoulder_pan for this experiment; other motors are undisturbed.
    motors={"shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
)
bus.connect()

# Read current position in raw ticks to avoid needing calibration.
start_ticks = bus.read("Present_Position", "shoulder_pan", normalize=False)
target_ticks = int(start_ticks + STEP_DEG * TICKS_PER_DEG)

print(f"Start : {start_ticks} ticks")
print(f"Target: {target_ticks} ticks  (+{STEP_DEG}°)")
print()
input("Make sure shoulder_pan has 30° of clear space, then press ENTER to command the step.")

times = []
positions_deg = []  # relative to start, in degrees

t0 = time.perf_counter()

# Send the step command once — the servo's internal controller does the rest.
bus.write("Goal_Position", "shoulder_pan", target_ticks, normalize=False)

# Sample Present_Position as fast as the bus allows (~100 Hz over USB serial).
while time.perf_counter() - t0 < LOG_DURATION_S:
    raw = bus.read("Present_Position", "shoulder_pan", normalize=False)
    times.append(time.perf_counter() - t0)
    positions_deg.append((raw - start_ticks) / TICKS_PER_DEG)

bus.disconnect()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(STEP_DEG, color="r", linestyle="--", linewidth=1, label=f"target ({STEP_DEG}°)")
ax.plot(times, positions_deg, linewidth=1.5, label="measured position")
ax.set_xlabel("time (s)")
ax.set_ylabel("position (°)")
ax.set_title("Step Response — shoulder_pan: 0° → 180°")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "step_response.png", dpi=150)
print(f"Saved {PLOTS_DIR / 'step_response.png'}")
