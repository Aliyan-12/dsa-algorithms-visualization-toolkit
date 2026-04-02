"""
Adam Optimizer Visualizer
=========================
Visualizes the Adam optimizer navigating a 2D loss surface in real-time.

Adam (Adaptive Moment Estimation) combines the benefits of RMSProp and
momentum by tracking both first and second moment estimates of gradients.

Update rule:
  m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          (1st moment — mean)
  v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²          (2nd moment — variance)
  m̂_t = m_t / (1 - β₁ᵗ)                          (bias correction)
  v̂_t = v_t / (1 - β₂ᵗ)                          (bias correction)
  θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

Time Complexity:  O(n) per step — n = number of parameters
Space Complexity: O(n) — stores m and v for each parameter

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
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
NUM_STEPS = 200
INTERVAL_MS = 50

# --- Loss surface: Rosenbrock-like function ---
def loss_fn(x, y):
    return (1 - x)**2 + 10 * (y - x**2)**2

def grad_fn(x, y):
    dx = -2 * (1 - x) - 40 * x * (y - x**2)
    dy = 20 * (y - x**2)
    return np.array([dx, dy])


def main():
    theta = np.array([-1.5, 2.0])
    m = np.zeros(2)  # first moment
    v = np.zeros(2)  # second moment

    path = [theta.copy()]
    losses = [loss_fn(*theta)]
    moment_norms = {"m": [0.0], "v": [0.0]}

    for t in range(1, NUM_STEPS + 1):
        g = grad_fn(*theta)
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g**2

        m_hat = m / (1 - BETA1**t)  # bias correction
        v_hat = v / (1 - BETA2**t)

        theta = theta - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)
        path.append(theta.copy())
        losses.append(loss_fn(*theta))
        moment_norms["m"].append(np.linalg.norm(m_hat))
        moment_norms["v"].append(np.linalg.norm(v_hat))

    path = np.array(path)

    # --- Plotting ---
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Adam Optimizer Visualization", fontsize=16,
                 color="white", fontweight="bold")

    ax_surface = fig.add_subplot(1, 3, (1, 2))
    ax_loss = fig.add_subplot(1, 3, 3)

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

    loss_line, = ax_loss.plot([], [], "-", color="#00b4d8", linewidth=2, label="Loss")
    ax_loss.set_xlim(0, NUM_STEPS)
    ax_loss.set_ylim(max(min(losses) * 0.5, 1e-6), losses[0] * 2)
    ax_loss.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)

    info_text = ax_surface.text(
        0.02, 0.95, "", transform=ax_surface.transAxes, fontsize=9,
        color="white", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", alpha=0.8),
    )
    param_text = ax_surface.text(
        0.98, 0.95,
        f"lr={LEARNING_RATE}  β₁={BETA1}  β₂={BETA2}\nε={EPSILON}\n"
        f"Time: O(n)/step\nSpace: O(n)",
        transform=ax_surface.transAxes, fontsize=9, color="#e0e0e0",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
    )

    # Moment annotation
    moment_text = ax_surface.text(
        0.02, 0.02, "", transform=ax_surface.transAxes, fontsize=8,
        color="#e0e0e0", verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#533483", alpha=0.8),
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
        moment_text.set_text(
            f"‖m̂‖ = {moment_norms['m'][i]:.4f}   ‖v̂‖ = {moment_norms['v'][i]:.4f}"
        )

        if i == NUM_STEPS:
            point.set_color("#00b4d8")
            point.set_markersize(10)

        return trail_line, point, loss_line, info_text, moment_text

    anim = FuncAnimation(
        fig, update, frames=range(NUM_STEPS + 1),
        interval=INTERVAL_MS, repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
