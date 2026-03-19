"""
Quick Sort Visualizer
=====================
Visualizes the quick sort algorithm step-by-step using matplotlib bar charts.

Time Complexity:  O(n log n) Average, O(n^2) Worst case
Space Complexity: O(log n) — In-place, stack space for recursion
Stable:           No

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ARRAY_SIZE = 50
INTERVAL_MS = 50


def quicksort_generator(arr):
    """Generator that yields array state after each comparison/swap."""
    yield from _quicksort(arr, 0, len(arr) - 1)


def _quicksort(arr, low, high):
    if low < high:
        yield from _partition_and_sort(arr, low, high)


def _partition_and_sort(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        yield arr, j, high, low, high  # comparing j with pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            yield arr, i, j, low, high  # swap happened

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    pivot_idx = i + 1
    yield arr, pivot_idx, -1, low, high  # pivot placed

    yield from _quicksort(arr, low, pivot_idx - 1)
    yield from _quicksort(arr, pivot_idx + 1, high)


def main():
    arr = [random.randint(1, 100) for _ in range(ARRAY_SIZE)]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("Quick Sort Visualization", fontsize=16, color="white", fontweight="bold")

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
    ax.text(
        0.98, 0.95,
        "Time: O(n log n) avg\n          O(n²) worst\nSpace: O(log n)\nStable: No",
        transform=ax.transAxes, fontsize=9, color="#e0e0e0",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
    )

    gen = quicksort_generator(arr)
    operations = [0]
    start_time = [time.time()]

    def update(frame):
        arr_state, idx1, idx2, low, high = frame
        operations[0] += 1

        for i, (bar, val) in enumerate(zip(bars, arr_state)):
            bar.set_height(val)
            if i == idx1:
                bar.set_color("#e94560")  # red — active element
            elif i == idx2:
                bar.set_color("#ffd460")  # yellow — pivot / second element
            elif low <= i <= high:
                bar.set_color("#533483")  # purple — active partition
            else:
                bar.set_color("#0f3460")  # default blue

        elapsed = time.time() - start_time[0]
        info_text.set_text(f"Operations: {operations[0]}  |  Time: {elapsed:.3f}s")
        return list(bars) + [info_text]

    anim = FuncAnimation(
        fig, update, frames=gen, interval=INTERVAL_MS,
        repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
