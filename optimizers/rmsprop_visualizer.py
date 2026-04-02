"""
RMSProp Optimizer Visualizer
=============================
Visualizes the RMSProp optimizer navigating a 2D loss surface in real-time.

RMSProp (Root Mean Square Propagation) adapts the learning rate for each
parameter by dividing by the running average of squared gradients.

Update rule:
  v_t = ρ * v_{t-1} + (1 - ρ) * g_t²
  θ_t = θ_{t-1} - (lr / √(v_t + ε)) * g_t

Time Complexity:  O(n) per step — n = number of parameters
Space Complexity: O(n) — stores running average v for each parameter

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# --- Optimizer hyperparameters ---
LEARNING_RATE = 0.05
DECAY_RATE = 0.9       # ρ — exponential decay of squared gradient average
EPSILON = 1e-8
NUM_STEPS = 200
INTERVAL_MS = 50

# --- Loss surface: Rosenbrock-like function (narrow curved valley) ---
def loss_fn(x, y):
    return (1 - x)**2 + 10 * (y - x**2)**2

def grad_fn(x, y):
    dx = -2 * (1 - x) - 40 * x * (y - x**2)
    dy = 20 * (y - x**2)
    return np.array([dx, dy])


def main():
    # Starting point
    theta = np.array([-1.5, 2.0])
    v = np.zeros(2)  # running avg of squared gradients

    # Pre-compute path
    path = [theta.copy()]
    losses = [loss_fn(*theta)]

    for _ in range(NUM_STEPS):
        g = grad_fn(*theta)
        v = DECAY_RATE * v + (1 - DECAY_RATE) * g**2
        theta = theta - (LEARNING_RATE / (np.sqrt(v) + EPSILON)) * g
        path.append(theta.copy())
        losses.append(loss_fn(*theta))

    path = np.array(path)

    # --- Plotting ---
    fig, (ax_surface, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("RMSProp Optimizer Visualization", fontsize=16,
                 color="white", fontweight="bold")

    # Contour plot
    ax_surface.set_facecolor("#16213e")
    xs = np.linspace(-2.5, 2.5, 300)
    ys = np.linspace(-1.5, 4.0, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = loss_fn(X, Y)

    levels = np.logspace(-1, 3.5, 25)
    ax_surface.contourf(X, Y, Z, levels=levels, cmap="inferno", alpha=0.8)
    ax_surface.contour(X, Y, Z, levels=levels, colors="white", alpha=0.15, linewidths=0.5)
    ax_surface.plot(1, 1, marker="*", markersize=15, color="#00b4d8", zorder=5)
    ax_surface.set_title("Loss Surface (Rosenbrock)", fontsize=11, color="white")
    ax_surface.tick_params(colors="white")
    for spine in ax_surface.spines.values():
        spine.set_color("#333")

    trail_line, = ax_surface.plot([], [], "-", color="#e94560", linewidth=1.5, alpha=0.7)
    point, = ax_surface.plot([], [], "o", color="#e94560", markersize=8, zorder=5)

    # Loss curve
    ax_loss.set_facecolor("#16213e")
    ax_loss.set_title("Loss Over Steps", fontsize=11, color="white")
    ax_loss.set_xlabel("Step", color="white")
    ax_loss.set_ylabel("Loss (log scale)", color="white")
    ax_loss.set_yscale("log")
    ax_loss.tick_params(colors="white")
    for spine in ax_loss.spines.values():
        spine.set_color("#333")

    loss_line, = ax_loss.plot([], [], "-", color="#00b4d8", linewidth=2)
    ax_loss.set_xlim(0, NUM_STEPS)
    ax_loss.set_ylim(max(min(losses) * 0.5, 1e-6), losses[0] * 2)

    info_text = ax_surface.text(
        0.02, 0.95, "", transform=ax_surface.transAxes, fontsize=9,
        color="white", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", alpha=0.8),
    )
    param_text = ax_surface.text(
        0.98, 0.95,
        f"lr={LEARNING_RATE}  ρ={DECAY_RATE}\nε={EPSILON}\n"
        f"Time: O(n)/step\nSpace: O(n)",
        transform=ax_surface.transAxes, fontsize=9, color="#e0e0e0",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
    )

    start_time = [time.time()]

    def update(frame):
        i = frame
        trail_line.set_data(path[:i+1, 0], path[:i+1, 1])
        point.set_data([path[i, 0]], [path[i, 1]])

        loss_line.set_data(range(i+1), losses[:i+1])

        elapsed = time.time() - start_time[0]
        info_text.set_text(
            f"Step: {i}/{NUM_STEPS}  |  Loss: {losses[i]:.6f}\n"
            f"θ = ({path[i,0]:.3f}, {path[i,1]:.3f})  |  Time: {elapsed:.2f}s"
        )

        if i == NUM_STEPS:
            point.set_color("#00b4d8")
            point.set_markersize(10)

        return trail_line, point, loss_line, info_text

    anim = FuncAnimation(
        fig, update, frames=range(NUM_STEPS + 1),
        interval=INTERVAL_MS, repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
