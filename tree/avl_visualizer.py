"""
AVL Tree (Self-Balancing BST) Visualizer
=========================================
Visualizes AVL tree insertion with automatic rotations step-by-step
using NetworkX and Matplotlib.

Time Complexity:
  - Search/Insert/Delete: O(log n) — guaranteed, due to balancing
Space Complexity: O(n)

Rotations:
  - Left Rotation:  when right-heavy
  - Right Rotation: when left-heavy
  - Left-Right:     left child is right-heavy
  - Right-Left:     right child is left-heavy

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

INTERVAL_MS = 1000


class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


def _height(node):
    return node.height if node else 0


def _balance_factor(node):
    return _height(node.left) - _height(node.right) if node else 0


def _update_height(node):
    if node:
        node.height = 1 + max(_height(node.left), _height(node.right))


def _rotate_right(y):
    x = y.left
    t2 = x.right
    x.right = y
    y.left = t2
    _update_height(y)
    _update_height(x)
    return x


def _rotate_left(x):
    y = x.right
    t2 = y.left
    y.left = x
    x.right = t2
    _update_height(x)
    _update_height(y)
    return y


def _compute_positions(node, x=0, y=0, dx=2.0, positions=None):
    if positions is None:
        positions = {}
    if node is None:
        return positions
    positions[node.key] = (x, y)
    _compute_positions(node.left, x - dx, y - 1, dx * 0.55, positions)
    _compute_positions(node.right, x + dx, y - 1, dx * 0.55, positions)
    return positions


def _build_nx_graph(node, G=None):
    if G is None:
        G = nx.DiGraph()
    if node is None:
        return G
    G.add_node(node.key)
    if node.left:
        G.add_edge(node.key, node.left.key)
        _build_nx_graph(node.left, G)
    if node.right:
        G.add_edge(node.key, node.right.key)
        _build_nx_graph(node.right, G)
    return G


def _get_balance_labels(node, labels=None):
    """Get balance factor for each node."""
    if labels is None:
        labels = {}
    if node is None:
        return labels
    labels[node.key] = _balance_factor(node)
    _get_balance_labels(node.left, labels)
    _get_balance_labels(node.right, labels)
    return labels


def avl_insert_generator(values):
    """Generator that yields tree state after each operation."""
    root = [None]  # mutable container

    def insert(node, key, path):
        if node is None:
            return AVLNode(key)

        path.append(node.key)
        if key < node.key:
            node.left = insert(node.left, key, path)
        elif key > node.key:
            node.right = insert(node.right, key, path)
        else:
            return node  # duplicate

        _update_height(node)
        bf = _balance_factor(node)

        # Left Left
        if bf > 1 and key < node.left.key:
            rotated = _rotate_right(node)
            return rotated

        # Right Right
        if bf < -1 and key > node.right.key:
            rotated = _rotate_left(node)
            return rotated

        # Left Right
        if bf > 1 and key > node.left.key:
            node.left = _rotate_left(node.left)
            rotated = _rotate_right(node)
            return rotated

        # Right Left
        if bf < -1 and key < node.right.key:
            node.right = _rotate_right(node.right)
            rotated = _rotate_left(node)
            return rotated

        return node

    for val in values:
        path = []
        root[0] = insert(root[0], val, path)
        path.append(val)

        balances = _get_balance_labels(root[0])
        has_rotation = any(abs(b) > 1 for b in balances.values())
        rotation_info = " (rotation applied!)" if has_rotation else ""

        yield root[0], val, path, balances, f"Insert {val}{rotation_info}"


def main():
    values = [30, 20, 40, 10, 25, 35, 50, 5, 15, 22, 27, 45, 60, 3, 8]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("AVL Tree Visualization", fontsize=16, color="white", fontweight="bold")
    ax.axis("off")

    gen = avl_insert_generator(values)
    step_count = [0]
    start_time = [time.time()]

    def update(frame):
        root, active_val, path, balances, msg = frame
        step_count[0] += 1
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axis("off")

        G = _build_nx_graph(root)
        pos = _compute_positions(root)

        if not G.nodes():
            return

        # Node colors
        node_colors = []
        for node in G.nodes():
            bf = balances.get(node, 0)
            if node == active_val:
                node_colors.append("#e94560")  # red — newly inserted
            elif abs(bf) > 1:
                node_colors.append("#ff6b35")  # orange — unbalanced
            elif node in path:
                node_colors.append("#ffd460")  # yellow — path
            else:
                node_colors.append("#0f3460")  # default blue

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800, edgecolors="white", linewidths=2)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", width=2, arrows=True, arrowsize=15)

        # Labels with balance factor
        labels = {n: f"{n}\nbf={balances.get(n, 0)}" for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_color="white", font_weight="bold", font_size=9)

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
            "Time: O(log n) guaranteed\nSpace: O(n)\nSelf-balancing BST",
            transform=ax.transAxes, fontsize=9, color="#e0e0e0",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
        )
        ax.set_title("AVL Tree Visualization", fontsize=16, color="white", fontweight="bold")

    anim = FuncAnimation(
        fig, update, frames=gen, interval=INTERVAL_MS,
        repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
