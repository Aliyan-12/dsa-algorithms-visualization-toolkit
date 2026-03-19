"""
Binary Search Tree (BST) Operations Visualizer
================================================
Visualizes BST insertion and search operations step-by-step
using NetworkX and Matplotlib.

Time Complexity:
  - Search/Insert: O(h) where h = height
  - Best case (balanced): O(log n)
  - Worst case (skewed):  O(n)
Space Complexity: O(n) for storage

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

INTERVAL_MS = 800


class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


def _compute_positions(node, x=0, y=0, dx=1.5, positions=None):
    """Compute tree node positions for drawing (top-down layout)."""
    if positions is None:
        positions = {}
    if node is None:
        return positions
    positions[node.key] = (x, y)
    _compute_positions(node.left, x - dx, y - 1, dx * 0.6, positions)
    _compute_positions(node.right, x + dx, y - 1, dx * 0.6, positions)
    return positions


def _build_nx_graph(node, G=None, edges=None):
    """Convert BST to NetworkX directed graph."""
    if G is None:
        G = nx.DiGraph()
        edges = []
    if node is None:
        return G, edges
    G.add_node(node.key)
    if node.left:
        G.add_edge(node.key, node.left.key)
        edges.append((node.key, node.left.key))
        _build_nx_graph(node.left, G, edges)
    if node.right:
        G.add_edge(node.key, node.right.key)
        edges.append((node.key, node.right.key))
        _build_nx_graph(node.right, G, edges)
    return G, edges


def bst_insert_generator(values):
    """Generator that yields tree state after each insertion step."""
    root = None

    for val in values:
        # Insertion traversal
        if root is None:
            root = BSTNode(val)
            yield root, val, [val], f"Insert {val} as root"
            continue

        path = []
        current = root
        while True:
            path.append(current.key)
            if val < current.key:
                if current.left is None:
                    current.left = BSTNode(val)
                    path.append(val)
                    yield root, val, path, f"Insert {val} (go left from {current.key})"
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = BSTNode(val)
                    path.append(val)
                    yield root, val, path, f"Insert {val} (go right from {current.key})"
                    break
                current = current.right

    # Search demonstration
    search_vals = [values[len(values) // 2], values[0], 999]
    for target in search_vals:
        path = []
        current = root
        found = False
        while current:
            path.append(current.key)
            yield root, target, path, f"Search {target}: visiting {current.key}"
            if target == current.key:
                found = True
                break
            elif target < current.key:
                current = current.left
            else:
                current = current.right

        result = "FOUND" if found else "NOT FOUND"
        yield root, target, path, f"Search {target}: {result}!"


def main():
    values = [50, 30, 70, 20, 40, 60, 80, 10, 35, 45, 65, 75, 55]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("BST Operations Visualization", fontsize=16, color="white", fontweight="bold")
    ax.axis("off")

    gen = bst_insert_generator(values)
    step_count = [0]
    start_time = [time.time()]

    def update(frame):
        root, active_val, path, msg = frame
        step_count[0] += 1
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axis("off")

        G, _ = _build_nx_graph(root)
        pos = _compute_positions(root)

        if not G.nodes():
            return

        # Node colors
        node_colors = []
        for node in G.nodes():
            if node == active_val and node == path[-1]:
                node_colors.append("#e94560")  # red — newly inserted/found
            elif node in path:
                node_colors.append("#ffd460")  # yellow — traversal path
            else:
                node_colors.append("#0f3460")  # default blue

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800, edgecolors="white", linewidths=2)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", width=2, arrows=True, arrowsize=15)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold", font_size=11)

        elapsed = time.time() - start_time[0]
        ax.text(
            0.02, 0.02, f"Step: {step_count[0]}  |  Nodes: {len(G.nodes())}  |  Time: {elapsed:.3f}s",
            transform=ax.transAxes, fontsize=10, color="white", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", alpha=0.8),
        )
        ax.text(
            0.5, 0.98, msg, transform=ax.transAxes, fontsize=11,
            color="#ffd460", verticalalignment="top", horizontalalignment="center",
            fontweight="bold",
        )
        ax.text(
            0.98, 0.02,
            "Search/Insert: O(log n) avg\n                     O(n) worst\nSpace: O(n)",
            transform=ax.transAxes, fontsize=9, color="#e0e0e0",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
        )
        ax.set_title("BST Operations Visualization", fontsize=16, color="white", fontweight="bold")

    anim = FuncAnimation(
        fig, update, frames=gen, interval=INTERVAL_MS,
        repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
