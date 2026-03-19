"""
Heap Sort Visualizer
====================
Visualizes the heap sort algorithm step-by-step using matplotlib bar charts.

Time Complexity:  O(n log n) — Best, Average, and Worst case
Space Complexity: O(1) — In-place sorting
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


def heapsort_generator(arr):
    """Generator that yields array state after each swap."""
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        yield from _heapify(arr, n, i, n)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        yield arr, 0, i, i  # swap root with end, sorted boundary
        yield from _heapify(arr, i, 0, i)


def _heapify(arr, heap_size, root, sorted_boundary):
    largest = root
    left = 2 * root + 1
    right = 2 * root + 2

    if left < heap_size and arr[left] > arr[largest]:
        largest = left
    if right < heap_size and arr[right] > arr[largest]:
        largest = right

    if largest != root:
        arr[root], arr[largest] = arr[largest], arr[root]
        yield arr, root, largest, sorted_boundary
        yield from _heapify(arr, heap_size, largest, sorted_boundary)


def main():
    arr = [random.randint(1, 100) for _ in range(ARRAY_SIZE)]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("Heap Sort Visualization", fontsize=16, color="white", fontweight="bold")

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
        "Time: O(n log n)\nSpace: O(1)\nStable: No",
        transform=ax.transAxes, fontsize=9, color="#e0e0e0",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
    )

    gen = heapsort_generator(arr)
    operations = [0]
    start_time = [time.time()]

    def update(frame):
        arr_state, idx1, idx2, sorted_boundary = frame
        operations[0] += 1

        for i, (bar, val) in enumerate(zip(bars, arr_state)):
            bar.set_height(val)
            if i >= sorted_boundary:
                bar.set_color("#00b4d8")  # cyan — sorted
            elif i == idx1:
                bar.set_color("#e94560")  # red — swapping
            elif i == idx2:
                bar.set_color("#ffd460")  # yellow — swapping
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
