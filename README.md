# Algorithm Visualizer

Interactive Python visualizations of fundamental algorithms — built as a learning resource for CS students.

Each script runs a **real-time animated visualization** using `matplotlib` and `networkx`, showing how the algorithm works step-by-step with live stats (operations count, runtime, time/space complexity).

## Algorithms

### Sorting Algorithms

| Algorithm  | File                                | Time                         | Space    |
| ---------- | ----------------------------------- | ---------------------------- | -------- |
| Merge Sort | `sorting/mergesort_visualizer.py` | O(n log n)                   | O(n)     |
| Quick Sort | `sorting/quicksort_visualizer.py` | O(n log n) avg, O(n²) worst | O(log n) |
| Heap Sort  | `sorting/heapsort_visualizer.py`  | O(n log n)                   | O(1)     |

### Graph Algorithms (Shortest Path)

| Algorithm              | File                                | Time           | Space  |
| ---------------------- | ----------------------------------- | -------------- | ------ |
| Dijkstra's Algorithm   | `graph/dijkstra_visualizer.py`    | O((V+E) log V) | O(V+E) |
| Bellman-Ford Algorithm | `graph/bellmanford_visualizer.py` | O(V·E)        | O(V)   |

### Tree & Data Structure Algorithms

| Algorithm      | File                       | Time                     | Space |
| -------------- | -------------------------- | ------------------------ | ----- |
| BST Operations | `tree/bst_visualizer.py` | O(log n) avg, O(n) worst | O(n)  |
| AVL Tree       | `tree/avl_visualizer.py` | O(log n) guaranteed      | O(n)  |

### Greedy Algorithms

| Algorithm      | File                             | Time       | Space |
| -------------- | -------------------------------- | ---------- | ----- |
| Huffman Coding | `greedy/huffman_visualizer.py` | O(n log n) | O(n)  |

## Setup

```bash
# Clone the repo
git clone https://github.com/Aliyan-12/dsa-algorithms-visualization-toolkit.git
cd algorithm-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run any visualizer directly:

```bash
python sorting/mergesort_visualizer.py
python sorting/quicksort_visualizer.py
python sorting/heapsort_visualizer.py
python graph/dijkstra_visualizer.py
python graph/bellmanford_visualizer.py
python tree/bst_visualizer.py
python tree/avl_visualizer.py
python greedy/huffman_visualizer.py
```

Each visualization window shows:

- **Real-time animation** of the algorithm in action
- **Color-coded elements** (active, comparing, sorted, etc.)
- **Operation counter** and **elapsed time**
- **Time & space complexity** reference

## Dependencies

- `matplotlib` — Charts, animations, and rendering
- `networkx` — Graph and tree data structures + drawing
- `numpy` — Array operations

## Project Structure

```
dsa-algorithms-visualization-toolkit/
├── sorting/
│   ├── mergesort_visualizer.py
│   ├── quicksort_visualizer.py
│   └── heapsort_visualizer.py
├── graph/
│   ├── dijkstra_visualizer.py
│   └── bellmanford_visualizer.py
├── tree/
│   ├── bst_visualizer.py
│   └── avl_visualizer.py
├── greedy/
│   └── huffman_visualizer.py
├── requirements.txt
└── README.md
```

## License

MIT — free to use for learning and teaching.
