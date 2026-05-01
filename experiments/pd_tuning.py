"""
Step 3: Outer-Loop PD Tuning
=============================
Runs a 100 Hz outer loop over the servo's internal position controller.
After each motion settles the plot updates automatically — no restart needed.

Tuning sequence:
  1. Start with Kp=0.01, Kd=0  →  arm barely moves (baseline)
  2. Raise Kp (e.g. kp 1.0, kp 3.0) until it oscillates
  3. Add Kd (e.g. kd 0.5) until oscillation dies
  4. Try a new target (e.g. target 45) to test tracking

Controls (type and press ENTER):
  kp <value>       set proportional gain      e.g.  kp 2.0
  kd <value>       set derivative gain        e.g.  kd 0.5
  target <value>   new target in degrees      e.g.  target 45
  save             save current plot to plots/
  q                quit

What to observe in the plot:
  - Kd=0, low Kp  →  slow approach, no overshoot
  - Kd=0, high Kp →  position oscillates around target
  - add Kd        →  oscillation dampens, velocity spike shrinks
  - too much Kd   →  arm moves sluggishly
"""

import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from drivers.feetech import FeetechMotorsBus
from drivers.motors_bus import Motor, MotorNormMode

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
PORT = "/dev/ttyACM0"
LOOP_HZ = 100
LATENCY_S = 0.050           # measured round-trip lag for position prediction

# Settle detection: position must stay within SETTLE_DEG of target
# for SETTLE_TICKS consecutive ticks before we consider the motion done.
SETTLE_DEG = 1.0
SETTLE_TICKS = 50           # 50 ticks @ 100 Hz = 0.5 s of stillness


TICKS_PER_DEG = 4096 / 360
# ──────────────────────────────────────────────────────────────────────────────

# ── Shared state (input thread → control thread) ──────────────────────────────
state = {
    "Kp": 0.25,
    "Kd": 0.01,
    "target_deg": 30.0,
    "running": True,
    "save_plot": False,
}
state_lock = threading.Lock()

# Signals the main thread that a motion just settled and the plot should update
settled_event = threading.Event()

# Written by control thread, read by main thread after settled_event fires.
# Holds the data from the most recent completed motion.
last_run: dict = {}


def input_thread():
    """Blocks on stdin, updates gains and target in shared state."""
    print("\n─── Controls ──────────────────────────────────────")
    print("  kp <val>      e.g.  kp 2.0")
    print("  kd <val>      e.g.  kd 0.5")
    print("  target <val>  e.g.  target 45")
    print("  save          save current plot to plots/")
    print("  q             quit")
    print("───────────────────────────────────────────────────\n")

    while True:
        try:
            line = input().strip().lower()
        except EOFError:
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        with state_lock:
            if cmd == "q":
                state["running"] = False
                # Unblock the main thread so it can exit cleanly
                settled_event.set()
                break
            elif cmd == "kp" and len(parts) == 2:
                state["Kp"] = float(parts[1])
                print(f"  → Kp = {state['Kp']}")
            elif cmd == "kd" and len(parts) == 2:
                state["Kd"] = float(parts[1])
                print(f"  → Kd = {state['Kd']}")
            elif cmd == "target" and len(parts) == 2:
                state["target_deg"] = float(parts[1])
                print(f"  → target = {state['target_deg']}°")
            elif cmd == "save":
                state["save_plot"] = True
            else:
                print(f"  unknown: '{line}'")


def control_thread(start_ticks: int):
    """
    100 Hz PD loop. Resets its log each time the target changes,
    then signals the main thread once the arm settles.
    """
    global last_run

    dt = 1.0 / LOOP_HZ
    t_start = time.perf_counter()

    # Per-motion log buffers — reset when target changes
    t_log, pos_log, vel_log, goal_log, target_log = [], [], [], [], []

    prev_target = None
    settle_count = 0    # consecutive ticks within SETTLE_DEG of target

    while True:
        t0 = time.perf_counter()

        with state_lock:
            if not state["running"]:
                break
            Kp = state["Kp"]
            Kd = state["Kd"]
            target_deg = state["target_deg"]

        # Reset log and settle counter whenever the target changes
        if target_deg != prev_target:
            t_log.clear(); pos_log.clear(); vel_log.clear()
            goal_log.clear(); target_log.clear()
            settle_count = 0
            t_start = time.perf_counter()
            prev_target = target_deg

        # ── Sense ─────────────────────────────────────────────────────────────
        raw_pos = bus.read("Present_Position", "shoulder_pan", normalize=False)
        raw_vel = bus.read("Present_Velocity", "shoulder_pan", normalize=False)

        pos_deg = (raw_pos - start_ticks) / TICKS_PER_DEG
        vel_deg_s = raw_vel / TICKS_PER_DEG

        # ── Predict ───────────────────────────────────────────────────────────
        # Extrapolate position forward by the measured round-trip lag so the
        # error term is computed on an estimate of where the arm is right now,
        # not where it was 50 ms ago.
        pos_predicted = pos_deg + vel_deg_s * LATENCY_S

        # ── PD law ────────────────────────────────────────────────────────────
        # Nudge Goal_Position rather than commanding torque directly.
        # Kp: pushes the goal further ahead the farther we are (aggression).
        # Kd: pulls the goal back proportional to velocity (braking).
        error = target_deg - pos_predicted
        goal_deg = target_deg + Kp * error - Kd * vel_deg_s

        goal_ticks = int(start_ticks + goal_deg * TICKS_PER_DEG)
        bus.write("Goal_Position", "shoulder_pan", goal_ticks, normalize=False)

        # ── Log ───────────────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        t_log.append(elapsed)
        pos_log.append(pos_deg)
        vel_log.append(vel_deg_s)
        goal_log.append(goal_deg)
        target_log.append(target_deg)

        # ── Settle detection ──────────────────────────────────────────────────
        # Count consecutive ticks near the target. Once we hit SETTLE_TICKS,
        # snapshot the log and wake the main thread to redraw the plot.
        if abs(pos_deg - target_deg) < SETTLE_DEG:
            settle_count += 1
        else:
            settle_count = 0

        if settle_count == SETTLE_TICKS:
            last_run = {
                "t":      list(t_log),
                "pos":    list(pos_log),
                "vel":    list(vel_log),
                "goal":   list(goal_log),
                "target": list(target_log),
                "Kp": Kp,
                "Kd": Kd,
            }
            settled_event.set()

        # ── Rate hold ─────────────────────────────────────────────────────────
        sleep_s = dt - (time.perf_counter() - t0)
        if sleep_s > 0:
            time.sleep(sleep_s)

    bus.disconnect()
    print("Disconnected.")


def update_plot(ax1, ax2, lines, data: dict):
    """Redraws both axes with data from the last completed motion."""
    t = np.array(data["t"])
    line_target, line_goal, line_pos, line_vel = lines

    line_target.set_data(t, data["target"])
    line_goal.set_data(t, data["goal"])
    line_pos.set_data(t, data["pos"])
    line_vel.set_data(t, data["vel"])

    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()
    ax1.set_title(f"PD Tuning — Kp={data['Kp']:.3f}  Kd={data['Kd']:.3f}")


def save_plot(data: dict, run_number: int):
    """Saves the last completed motion to a PNG."""
    t = np.array(data["t"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, data["target"], "r--", linewidth=1, label="target")
    ax1.plot(t, data["goal"], color="orange", linewidth=0.8, alpha=0.6, label="commanded goal")
    ax1.plot(t, data["pos"], linewidth=1.5, label="measured position")
    ax1.set_ylabel("position (°)")
    ax1.set_title(f"PD run {run_number} — Kp={data['Kp']:.3f}  Kd={data['Kd']:.3f}")
    ax1.legend(); ax1.grid(True, alpha=0.4)

    ax2.plot(t, data["vel"], linewidth=1, color="steelblue", label="velocity")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("velocity (°/s)")
    ax2.set_xlabel("time (s)")
    ax2.legend(); ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = PLOTS_DIR / f"pd_tuning_run{run_number}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


# ── Setup ─────────────────────────────────────────────────────────────────────
bus = FeetechMotorsBus(
    port=PORT,
    motors={"shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
)
bus.connect()

start_ticks = bus.read("Present_Position", "shoulder_pan", normalize=False)
print(f"Start position: {start_ticks} ticks")

threading.Thread(target=input_thread, daemon=True).start()
threading.Thread(target=control_thread, args=(start_ticks,), daemon=True).start()

# ── Plot setup (main thread) ──────────────────────────────────────────────────
# matplotlib GUI must live on the main thread on most Linux backends.
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
fig.tight_layout(pad=2.5)

line_target, = ax1.plot([], [], "r--", linewidth=1, label="target")
line_goal,   = ax1.plot([], [], color="orange", linewidth=0.8, alpha=0.6, label="commanded goal")
line_pos,    = ax1.plot([], [], linewidth=1.5, label="position")
ax1.set_ylabel("position (°)")
ax1.set_title("PD Tuning — waiting for first motion...")
ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.4)

line_vel, = ax2.plot([], [], linewidth=1, color="steelblue", label="velocity")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.set_ylabel("velocity (°/s)")
ax2.set_xlabel("time (s)")
ax2.legend(loc="upper left"); ax2.grid(True, alpha=0.4)

plt.show()

lines = (line_target, line_goal, line_pos, line_vel)
run_number = 0

# Main thread waits for the control thread to signal settle, then redraws.
while True:
    settled_event.wait()     # block until motion settles (or q is pressed)
    settled_event.clear()

    with state_lock:
        running = state["running"]
        do_save = state["save_plot"]
        if do_save:
            state["save_plot"] = False

    if not running:
        break

    if last_run:
        run_number += 1
        update_plot(ax1, ax2, lines, last_run)
        fig.canvas.draw()
        fig.canvas.flush_events()

        if do_save:
            save_plot(last_run, run_number)

plt.close("all")
