"""
AdamW Optimizer Visualizer
==========================
Visualizes AdamW navigating a 2D loss surface, with a side-by-side comparison
to standard Adam showing the effect of decoupled weight decay.

AdamW fixes a subtle flaw in Adam's L2 regularization by decoupling
weight decay from the adaptive gradient step.

Update rule:
  m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
  v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
  m̂_t = m_t / (1 - β₁ᵗ)
  v̂_t = v_t / (1 - β₂ᵗ)
  θ_t = (1 - λ·lr) * θ_{t-1} - lr * m̂_t / (√v̂_t + ε)   ← decoupled decay

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

# --- Hyperparameters ---
LEARNING_RATE = 0.05
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 0.01
NUM_STEPS = 200
INTERVAL_MS = 50

# --- Loss surface: Rosenbrock-like function ---
def loss_fn(x, y):
    return (1 - x)**2 + 10 * (y - x**2)**2

def grad_fn(x, y):
    dx = -2 * (1 - x) - 40 * x * (y - x**2)
    dy = 20 * (y - x**2)
    return np.array([dx, dy])


def run_adam(theta_init, num_steps):
    """Standard Adam with L2 regularization (weight decay in gradient)."""
    theta = theta_init.copy()
    m, v = np.zeros(2), np.zeros(2)
    path, losses = [theta.copy()], [loss_fn(*theta)]

    for t in range(1, num_steps + 1):
        g = grad_fn(*theta) + WEIGHT_DECAY * theta  # L2 added to gradient
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g**2
        m_hat = m / (1 - BETA1**t)
        v_hat = v / (1 - BETA2**t)
        theta = theta - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)
        path.append(theta.copy())
        losses.append(loss_fn(*theta))

    return np.array(path), losses


def run_adamw(theta_init, num_steps):
    """AdamW with decoupled weight decay."""
    theta = theta_init.copy()
    m, v = np.zeros(2), np.zeros(2)
    path, losses = [theta.copy()], [loss_fn(*theta)]
    weight_norms = [np.linalg.norm(theta)]

    for t in range(1, num_steps + 1):
        g = grad_fn(*theta)  # gradient WITHOUT weight decay
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g**2
        m_hat = m / (1 - BETA1**t)
        v_hat = v / (1 - BETA2**t)

        # Decoupled weight decay — applied directly to parameters
        theta = (1 - WEIGHT_DECAY * LEARNING_RATE) * theta \
                - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)

        path.append(theta.copy())
        losses.append(loss_fn(*theta))
        weight_norms.append(np.linalg.norm(theta))

    return np.array(path), losses, weight_norms


def main():
    theta_init = np.array([-1.5, 2.0])

    adam_path, adam_losses = run_adam(theta_init, NUM_STEPS)
    adamw_path, adamw_losses, adamw_wnorms = run_adamw(theta_init, NUM_STEPS)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("AdamW vs Adam — Decoupled Weight Decay", fontsize=16,
                 color="white", fontweight="bold")

    ax_surface, ax_loss, ax_wnorm = axes

    # --- Contour plot with both paths ---
    ax_surface.set_facecolor("#16213e")
    xs = np.linspace(-2.5, 2.5, 300)
    ys = np.linspace(-1.5, 4.0, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = loss_fn(X, Y)

    levels = np.logspace(-1, 3.5, 25)
    ax_surface.contourf(X, Y, Z, levels=levels, cmap="inferno", alpha=0.8)
    ax_surface.contour(X, Y, Z, levels=levels, colors="white", alpha=0.15, linewidths=0.5)
    ax_surface.plot(1, 1, marker="*", markersize=15, color="#00b4d8", zorder=5)
    ax_surface.set_title("Loss Surface — Path Comparison", fontsize=11, color="white")
    ax_surface.tick_params(colors="white")
    for spine in ax_surface.spines.values():
        spine.set_color("#333")

    # Adam trail (blue)
    adam_trail, = ax_surface.plot([], [], "-", color="#5c7cfa", linewidth=1.5,
                                  alpha=0.7, label="Adam + L2")
    adam_pt, = ax_surface.plot([], [], "o", color="#5c7cfa", markersize=7, zorder=5)

    # AdamW trail (red)
    adamw_trail, = ax_surface.plot([], [], "-", color="#e94560", linewidth=1.5,
                                    alpha=0.7, label="AdamW")
    adamw_pt, = ax_surface.plot([], [], "o", color="#e94560", markersize=7, zorder=5)

    ax_surface.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white",
                      fontsize=9, loc="lower right")

    # --- Loss comparison ---
    ax_loss.set_facecolor("#16213e")
    ax_loss.set_title("Loss Comparison", fontsize=11, color="white")
    ax_loss.set_xlabel("Step", color="white")
    ax_loss.set_ylabel("Loss (log scale)", color="white")
    ax_loss.set_yscale("log")
    ax_loss.tick_params(colors="white")
    for spine in ax_loss.spines.values():
        spine.set_color("#333")

    adam_loss_line, = ax_loss.plot([], [], "-", color="#5c7cfa", linewidth=2, label="Adam + L2")
    adamw_loss_line, = ax_loss.plot([], [], "-", color="#e94560", linewidth=2, label="AdamW")
    ax_loss.set_xlim(0, NUM_STEPS)
    all_losses = adam_losses + adamw_losses
    ax_loss.set_ylim(max(min(all_losses) * 0.5, 1e-6), max(all_losses) * 2)
    ax_loss.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", fontsize=8)

    # --- Weight norm plot (AdamW signature feature) ---
    ax_wnorm.set_facecolor("#16213e")
    ax_wnorm.set_title("Weight Norm (‖θ‖) — AdamW", fontsize=11, color="white")
    ax_wnorm.set_xlabel("Step", color="white")
    ax_wnorm.set_ylabel("‖θ‖", color="white")
    ax_wnorm.tick_params(colors="white")
    for spine in ax_wnorm.spines.values():
        spine.set_color("#333")

    wnorm_line, = ax_wnorm.plot([], [], "-", color="#e94560", linewidth=2)
    ax_wnorm.set_xlim(0, NUM_STEPS)
    ax_wnorm.set_ylim(0, max(adamw_wnorms) * 1.2)

    info_text = ax_surface.text(
        0.02, 0.95, "", transform=ax_surface.transAxes, fontsize=9,
        color="white", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", alpha=0.8),
    )
    param_text = ax_surface.text(
        0.98, 0.02,
        f"lr={LEARNING_RATE}  β₁={BETA1}  β₂={BETA2}\n"
        f"λ={WEIGHT_DECAY}  ε={EPSILON}\nTime: O(n)/step  Space: O(n)",
        transform=ax_surface.transAxes, fontsize=8, color="#e0e0e0",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.8),
    )

    start_time = [time.time()]

    def update(frame):
        i = frame

        adam_trail.set_data(adam_path[:i+1, 0], adam_path[:i+1, 1])
        adam_pt.set_data([adam_path[i, 0]], [adam_path[i, 1]])
        adamw_trail.set_data(adamw_path[:i+1, 0], adamw_path[:i+1, 1])
        adamw_pt.set_data([adamw_path[i, 0]], [adamw_path[i, 1]])

        adam_loss_line.set_data(range(i+1), adam_losses[:i+1])
        adamw_loss_line.set_data(range(i+1), adamw_losses[:i+1])

        wnorm_line.set_data(range(i+1), adamw_wnorms[:i+1])

        elapsed = time.time() - start_time[0]
        info_text.set_text(
            f"Step: {i}/{NUM_STEPS}  |  Time: {elapsed:.2f}s\n"
            f"Adam  Loss: {adam_losses[i]:.6f}\n"
            f"AdamW Loss: {adamw_losses[i]:.6f}"
        )

        if i == NUM_STEPS:
            adam_pt.set_color("#00b4d8")
            adamw_pt.set_color("#00b4d8")

        return (adam_trail, adam_pt, adamw_trail, adamw_pt,
                adam_loss_line, adamw_loss_line, wnorm_line, info_text)

    anim = FuncAnimation(
        fig, update, frames=range(NUM_STEPS + 1),
        interval=INTERVAL_MS, repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
