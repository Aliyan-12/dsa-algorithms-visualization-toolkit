"""
Dijkstra's Algorithm Visualizer
================================
Visualizes Dijkstra's shortest path algorithm on a weighted graph
using NetworkX and Matplotlib.

Time Complexity:  O((V + E) log V) with min-heap
Space Complexity: O(V + E)

How it works:
  1. Starts from a source node
  2. Greedily selects the unvisited node with smallest tentative distance
  3. Relaxes all edges from the current node
  4. Repeats until all nodes are visited

Note: Does NOT work with negative edge weights.

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import time
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

INTERVAL_MS = 800


def build_sample_graph():
    """Create a sample weighted graph for demonstration."""
    G = nx.Graph()
    edges = [
        ("A", "B", 4), ("A", "C", 2),
        ("B", "C", 1), ("B", "D", 5),
        ("C", "D", 8), ("C", "E", 10),
        ("D", "E", 2), ("D", "F", 6),
        ("E", "F", 3), ("E", "G", 4),
        ("F", "G", 1), ("B", "F", 12),
        ("A", "G", 15),
    ]
    G.add_weighted_edges_from(edges)
    return G, "A"


def dijkstra_generator(G, source):
    """Generator that yields algorithm state at each step."""
    dist = {node: float("inf") for node in G.nodes()}
    dist[source] = 0
    prev = {node: None for node in G.nodes()}
    visited = set()
    pq = [(0, source)]

    yield dist.copy(), visited.copy(), None, set(), "Initialize: set source distance to 0"

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue

        visited.add(u)
        yield dist.copy(), visited.copy(), u, set(), f"Visit node {u} (distance={d})"

        relaxed_edges = set()
        for v in G.neighbors(u):
            if v not in visited:
                weight = G[u][v]["weight"]
                new_dist = dist[u] + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))
                    relaxed_edges.add((u, v))

        if relaxed_edges:
            yield dist.copy(), visited.copy(), u, relaxed_edges, f"Relax edges from {u}"

    yield dist.copy(), visited.copy(), None, set(), "Algorithm complete!"


def main():
    G, source = build_sample_graph()
    pos = nx.spring_layout(G, seed=42, k=2)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("Dijkstra's Algorithm Visualization", fontsize=16, color="white", fontweight="bold")
    ax.axis("off")

    info_text = ax.text(
        0.02, 0.02, "", transform=ax.transAxes, fontsize=10,
        color="white", verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", alpha=0.8),
    )
    step_text = ax.text(
        0.5, 0.98, "", transform=ax.transAxes, fontsize=11,
        color="#ffd460", verticalalignment="top", horizontalalignment="center",
        fontweight="bold",
    )
    ax.text(
        0.98, 0.02,
        "Time: O((V+E) log V)\nSpace: O(V+E)\nNo negative weights",
        transform=ax.transAxes, fontsize=9, color="#e0e0e0",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
    )

    edge_labels = nx.get_edge_attributes(G, "weight")
    gen = dijkstra_generator(G, source)
    step_count = [0]
    start_time = [time.time()]

    def update(frame):
        dist, visited, current, relaxed, msg = frame
        step_count[0] += 1
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axis("off")

        # Node colors
        node_colors = []
        for node in G.nodes():
            if node == current:
                node_colors.append("#e94560")  # red — processing
            elif node in visited:
                node_colors.append("#00b4d8")  # cyan — visited
            else:
                node_colors.append("#555555")  # grey — unvisited

        # Edge colors
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            if (u, v) in relaxed or (v, u) in relaxed:
                edge_colors.append("#ffd460")  # yellow — relaxed
                edge_widths.append(3.0)
            else:
                edge_colors.append("#444444")
                edge_widths.append(1.5)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=700, edgecolors="white", linewidths=2)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold", font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_color="#aaaaaa", font_size=9)

        # Distance labels below nodes
        dist_labels = {n: f"d={dist[n]}" if dist[n] != float("inf") else "d=inf" for n in G.nodes()}
        label_pos = {n: (pos[n][0], pos[n][1] - 0.12) for n in G.nodes()}
        nx.draw_networkx_labels(G, label_pos, labels=dist_labels, ax=ax, font_color="#ffd460", font_size=8)

        elapsed = time.time() - start_time[0]
        ax.text(
            0.02, 0.02, f"Step: {step_count[0]}  |  Time: {elapsed:.3f}s",
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
            "Time: O((V+E) log V)\nSpace: O(V+E)\nNo negative weights",
            transform=ax.transAxes, fontsize=9, color="#e0e0e0",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
        )
        ax.set_title("Dijkstra's Algorithm Visualization", fontsize=16, color="white", fontweight="bold")

    anim = FuncAnimation(
        fig, update, frames=gen, interval=INTERVAL_MS,
        repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
