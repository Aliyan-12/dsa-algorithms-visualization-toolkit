"""
Merge Sort Visualizer
=====================
Visualizes the merge sort algorithm step-by-step using matplotlib bar charts.

Time Complexity:  O(n log n) — Best, Average, and Worst case
Space Complexity: O(n) — Requires auxiliary array for merging
Stable:           Yes

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ARRAY_SIZE = 50
INTERVAL_MS = 30  # milliseconds between frames


def merge_sort_generator(arr):
    """Generator that yields array state after each comparison/write."""
    yield from _merge_sort(arr, 0, len(arr) - 1)


def _merge_sort(arr, left, right):
    if left >= right:
        return
    mid = (left + right) // 2
    yield from _merge_sort(arr, left, mid)
    yield from _merge_sort(arr, mid + 1, right)
    yield from _merge(arr, left, mid, right)


def _merge(arr, left, mid, right):
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]
    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1
        yield arr, left, right, k - 1  # current write position

    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1
        yield arr, left, right, k - 1

    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1
        yield arr, left, right, k - 1


def main():
    arr = [random.randint(1, 100) for _ in range(ARRAY_SIZE)]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("Merge Sort Visualization", fontsize=16, color="white", fontweight="bold")

    bars = ax.bar(range(len(arr)), arr, color="#0f3460", edgecolor="none", width=0.8)
    ax.set_xlim(-1, len(arr))
    ax.set_ylim(0, 110)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    info_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
        color="white", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", alpha=0.8),
    )
    complexity_text = ax.text(
        0.98, 0.95,
        "Time: O(n log n)\nSpace: O(n)\nStable: Yes",
        transform=ax.transAxes, fontsize=9, color="#e0e0e0",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
    )

    gen = merge_sort_generator(arr)
    operations = [0]
    start_time = [time.time()]

    def update(frame):
        arr_state, left, right, current = frame
        operations[0] += 1

        for i, (bar, val) in enumerate(zip(bars, arr_state)):
            bar.set_height(val)
            if i == current:
                bar.set_color("#e94560")  # red — current write
            elif left <= i <= right:
                bar.set_color("#533483")  # purple — active merge range
            else:
                bar.set_color("#0f3460")  # default blue

        elapsed = time.time() - start_time[0]
        info_text.set_text(f"Operations: {operations[0]}  |  Time: {elapsed:.3f}s")
        return list(bars) + [info_text]

    anim = FuncAnimation(
        fig, update, frames=gen, interval=INTERVAL_MS,
        repeat=False, cache_frame_data=False,
    )

    def on_finish(i=None):
        for bar, val in zip(bars, arr):
            bar.set_height(val)
            bar.set_color("#00b4d8")
        elapsed = time.time() - start_time[0]
        info_text.set_text(f"Done! Operations: {operations[0]}  |  Time: {elapsed:.3f}s")
        fig.canvas.draw()

    anim.event_source.add_callback(lambda: None)  # keep reference alive
    fig.canvas.mpl_connect("close_event", lambda e: None)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
