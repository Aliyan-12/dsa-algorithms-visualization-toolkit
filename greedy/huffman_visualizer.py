"""
Huffman Coding Visualizer
==========================
Visualizes Huffman tree construction and encoding step-by-step
using NetworkX and Matplotlib.

Time Complexity:  O(n log n) — n = number of unique characters
Space Complexity: O(n)

How it works:
  1. Count character frequencies
  2. Create leaf nodes for each character
  3. Repeatedly merge the two lowest-frequency nodes
  4. The resulting tree defines optimal prefix-free codes

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import time
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

INTERVAL_MS = 1200


class HuffmanNode:
    _counter = 0  # tiebreaker for heapq

    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        self._id = HuffmanNode._counter
        HuffmanNode._counter += 1

    def __lt__(self, other):
        if self.freq == other.freq:
            return self._id < other._id
        return self.freq < other.freq

    @property
    def label(self):
        if self.char:
            return f"'{self.char}'\n{self.freq}"
        return str(self.freq)

    @property
    def node_id(self):
        return f"{self.char or 'int'}_{self._id}"


def _build_nx_tree(node, G=None, labels=None, parent_id=None, edge_label=None, edge_labels=None):
    """Convert Huffman tree to NetworkX graph."""
    if G is None:
        G = nx.DiGraph()
        labels = {}
        edge_labels = {}
    if node is None:
        return G, labels, edge_labels

    nid = node.node_id
    G.add_node(nid)
    labels[nid] = node.label

    if parent_id is not None:
        G.add_edge(parent_id, nid)
        edge_labels[(parent_id, nid)] = edge_label

    _build_nx_tree(node.left, G, labels, nid, "0", edge_labels)
    _build_nx_tree(node.right, G, labels, nid, "1", edge_labels)
    return G, labels, edge_labels


def _compute_positions(node, x=0, y=0, dx=2.5, positions=None):
    if positions is None:
        positions = {}
    if node is None:
        return positions
    positions[node.node_id] = (x, y)
    _compute_positions(node.left, x - dx, y - 1.2, dx * 0.55, positions)
    _compute_positions(node.right, x + dx, y - 1.2, dx * 0.55, positions)
    return positions


def _get_codes(node, prefix="", codes=None):
    """Extract Huffman codes from the tree."""
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.char:  # leaf
        codes[node.char] = prefix or "0"
    _get_codes(node.left, prefix + "0", codes)
    _get_codes(node.right, prefix + "1", codes)
    return codes


def huffman_generator(text):
    """Generator that yields tree state at each merge step."""
    # Count frequencies
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    # Create initial heap
    heap = [HuffmanNode(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)

    yield None, heap[:], f"Initial: {len(heap)} characters with frequencies"

    step = 0
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)
        step += 1

        yield merged, heap[:], f"Step {step}: merge freq {left.freq} + {right.freq} = {merged.freq}"

    root = heap[0]
    codes = _get_codes(root)
    code_str = "  ".join(f"'{c}'={code}" for c, code in sorted(codes.items()))
    yield root, heap[:], f"Complete! Codes: {code_str}"


def main():
    text = "huffman coding algorithm visualization"

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("Huffman Coding Visualization", fontsize=16, color="white", fontweight="bold")
    ax.axis("off")

    gen = huffman_generator(text)
    step_count = [0]
    start_time = [time.time()]

    def update(frame):
        root, heap, msg = frame
        step_count[0] += 1
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axis("off")

        if root is not None:
            G, labels, edge_labels = _build_nx_tree(root)
            pos = _compute_positions(root)

            # Node colors — leaves are cyan, internal are blue
            node_colors = []
            for nid in G.nodes():
                if G.out_degree(nid) == 0:  # leaf
                    node_colors.append("#00b4d8")
                else:
                    node_colors.append("#0f3460")

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=900, edgecolors="white", linewidths=2)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", width=2, arrows=True, arrowsize=12)
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_color="white", font_weight="bold", font_size=8)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_color="#ffd460", font_size=10, font_weight="bold")
        else:
            # Show initial frequencies
            freq = {}
            for item in heap:
                freq[item.char] = item.freq
            freq_str = "\n".join(f"  '{c}': {f}" for c, f in sorted(freq.items(), key=lambda x: -x[1]))
            ax.text(
                0.5, 0.5, f"Character Frequencies:\n{freq_str}",
                transform=ax.transAxes, fontsize=12, color="white",
                verticalalignment="center", horizontalalignment="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=1", facecolor="#0f3460", alpha=0.9),
            )

        elapsed = time.time() - start_time[0]
        ax.text(
            0.02, 0.02,
            f"Step: {step_count[0]}  |  Nodes in heap: {len(heap)}  |  Time: {elapsed:.3f}s",
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
            "Time: O(n log n)\nSpace: O(n)\nGreedy / Optimal prefix codes",
            transform=ax.transAxes, fontsize=9, color="#e0e0e0",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
        )
        ax.set_title("Huffman Coding Visualization", fontsize=16, color="white", fontweight="bold")

    anim = FuncAnimation(
        fig, update, frames=gen, interval=INTERVAL_MS,
        repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
