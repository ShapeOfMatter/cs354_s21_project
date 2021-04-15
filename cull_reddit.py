from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
import sys
from typing import cast, Callable, Dict, List, Mapping, NewType, Tuple, TypeVar

EDGES_FILENAME = 'datasets/REDDIT-MULTI-12K/REDDIT-MULTI-12K_A.txt'
NODE_GRAPHS_FILENAME = 'datasets/REDDIT-MULTI-12K/REDDIT-MULTI-12K_graph_indicator.txt'
GRAPH_LABELS_FILENAME = 'datasets/REDDIT-MULTI-12K/REDDIT-MULTI-12K_graph_labels.txt'
NEW_EDGES_FILENAME = 'datasets/REDDIT-MULTI-12K/REDDIT-MULTI-12K_NEW.txt'

Index = NewType('Index', int)
Node = NewType('Node', int)
Graph = NewType('Graph', int)

L = TypeVar('L')
M = TypeVar('M')
def for_file(filename: str, line_map: Callable[[Index, str], L], reduction: Callable[[M, L], M], init: M) -> M:
    with open(filename) as f:
        value = init
        for i, l in enumerate(f, start=1):
            value = reduction(value, line_map(Index(i), l))
        return value

def main(threshold: int):
    num_graphs: int = for_file(GRAPH_LABELS_FILENAME, lambda i, l: i, lambda a, l: l, 0)
    print(f'counted {num_graphs} graphs.')
    def reduction(sizes_homes: Tuple[Dict[Graph, int], Dict[Node, Graph]], node_graph: Tuple[Node, Graph]):
        sizes_homes[0][node_graph[1]] += 1
        sizes_homes[1][node_graph[0]] = node_graph[1]
        return sizes_homes
    graph_sizes: Mapping[Graph, int]
    node_homes: Mapping[Node, Graph]
    graph_sizes, node_homes = for_file(NODE_GRAPHS_FILENAME,
                                       lambda i, l: (Node(i), Graph(int(l))),
                                       reduction,
                                       (cast(Dict[Graph, int], defaultdict(lambda: 0)),
                                        cast(Dict[Node, Graph], {})))
    plt.hist(graph_sizes.values(), bins=100)
    plt.show()
    with open(NEW_EDGES_FILENAME, 'w') as output:
        def record(saved_skipped: List[int], edge: Tuple[Node, ...]) -> List[int]:
            assert len(edge) == 2, f'{edge} does not seem to be an edge!'
            if any(graph_sizes[node_homes[node]] > threshold for node in edge):
                output.write(', '.join(str(e) for e in edge) + '\n')
                saved_skipped[0] += 1
            else:
                saved_skipped[1] += 1
            return saved_skipped
        saved, skipped = for_file(EDGES_FILENAME, lambda i, l: tuple(Node(int(n.strip())) for n in l.split(',')), record, [0, 0])
        print(f'Saved {saved} edges and skipped {skipped} edges.')


if __name__ == "__main__":
    main(int(sys.argv[1]))
