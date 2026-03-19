"""
Bellman-Ford Algorithm Visualizer
==================================
Visualizes the Bellman-Ford shortest path algorithm on a weighted directed graph
using NetworkX and Matplotlib.

Time Complexity:  O(V * E)
Space Complexity: O(V)

How it works:
  1. Initialize distances: source=0, all others=infinity
  2. Relax ALL edges V-1 times
  3. Check for negative weight cycles on the Vth pass

Advantage over Dijkstra: Handles negative edge weights.

Controls:
  - Close the window to exit
  - The animation runs automatically on launch
"""

import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

INTERVAL_MS = 600


def build_sample_graph():
    """Create a sample weighted directed graph (with some negative edges)."""
    G = nx.DiGraph()
    edges = [
        ("S", "A", 6), ("S", "B", 7),
        ("A", "C", 5), ("A", "D", -4),
        ("B", "A", 8), ("B", "E", -3),
        ("C", "B", -2), ("D", "C", 7),
        ("D", "A", 9), ("E", "D", 2),
        ("E", "C", 4),
    ]
    G.add_weighted_edges_from(edges)
    return G, "S"


def bellman_ford_generator(G, source):
    """Generator that yields algorithm state at each relaxation step."""
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    dist = {node: float("inf") for node in nodes}
    dist[source] = 0

    yield dist.copy(), set(), None, "Initialize: source distance = 0"

    n = len(nodes)
    for iteration in range(1, n):
        relaxed_this_round = set()
        for u, v, data in edges:
            w = data["weight"]
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                relaxed_this_round.add((u, v))
                yield dist.copy(), relaxed_this_round, (u, v), \
                    f"Iteration {iteration}: relax ({u}->{v}), d[{v}]={dist[v]}"

        if not relaxed_this_round:
            yield dist.copy(), set(), None, f"Iteration {iteration}: no relaxations — done early!"
            break
        else:
            yield dist.copy(), relaxed_this_round, None, f"Iteration {iteration} complete"

    # Check for negative cycles
    has_neg_cycle = False
    for u, v, data in edges:
        w = data["weight"]
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            has_neg_cycle = True
            break

    msg = "NEGATIVE CYCLE DETECTED!" if has_neg_cycle else "Algorithm complete! No negative cycles."
    yield dist.copy(), set(), None, msg


def main():
    G, source = build_sample_graph()
    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_title("Bellman-Ford Algorithm Visualization", fontsize=16, color="white", fontweight="bold")
    ax.axis("off")

    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    gen = bellman_ford_generator(G, source)
    step_count = [0]
    start_time = [time.time()]

    def update(frame):
        dist, relaxed_edges, current_edge, msg = frame
        step_count[0] += 1
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axis("off")

        # Node colors
        node_colors = []
        for node in G.nodes():
            if dist[node] == float("inf"):
                node_colors.append("#555555")
            elif current_edge and node == current_edge[1]:
                node_colors.append("#e94560")  # red — just updated
            elif dist[node] < float("inf"):
                node_colors.append("#00b4d8")  # cyan — reachable
            else:
                node_colors.append("#555555")

        # Edge colors
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            if (u, v) in relaxed_edges:
                edge_colors.append("#ffd460")
                edge_widths.append(3.0)
            else:
                edge_colors.append("#444444")
                edge_widths.append(1.5)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=700, edgecolors="white", linewidths=2)
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1",
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold", font_size=12)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_color="#aaaaaa", font_size=9)

        # Distance labels
        dist_labels = {n: f"d={dist[n]}" if dist[n] != float("inf") else "d=inf" for n in G.nodes()}
        label_pos = {n: (pos[n][0], pos[n][1] - 0.15) for n in G.nodes()}
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
            "Time: O(V * E)\nSpace: O(V)\nHandles negative weights",
            transform=ax.transAxes, fontsize=9, color="#e0e0e0",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", alpha=0.8),
        )
        ax.set_title("Bellman-Ford Algorithm Visualization", fontsize=16, color="white", fontweight="bold")

    anim = FuncAnimation(
        fig, update, frames=gen, interval=INTERVAL_MS,
        repeat=False, cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
