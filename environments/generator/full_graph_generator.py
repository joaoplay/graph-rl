from copy import deepcopy

import networkx as nx

from environments.generator.graph_node import GraphNode
from environments.generator.utils import id_generator
from graphs.graph_state import GraphState


class FullGraphGenerator:

    def __init__(self, size_x, size_y, interval_between_nodes) -> None:
        super().__init__()
        self.size_x = size_x
        self.size_y = size_y
        self.interval_between_nodes = interval_between_nodes

    def generate_nx_graph(self):
        graph = nx.Graph()

        id_gen = id_generator()

        adjacency_list = []

        nodes_list: list[GraphNode] = []
        previous_row_nodes = None
        current_row_nodes: list[GraphNode] = []
        for x in range(0, self.size_x, self.interval_between_nodes):
            for y in range(0, self.size_y, self.interval_between_nodes):
                current_node = GraphNode(unique_id=next(id_gen), coordinates=(x, y))
                current_row_nodes += [current_node]
                nodes_list += [current_node]

                if y > 0:
                    # Connect current node to the previous one at the same row
                    previous_node = current_row_nodes[y - 1]
                    adjacency_list += [(previous_node.unique_id, current_node.unique_id),
                                       (current_node.unique_id, previous_node.unique_id)]

                if previous_row_nodes is not None:
                    # Connect the current node to the neighbour in the previous row
                    min_y = max(0, y - 1)
                    max_y = min(y + 1, self.size_y - 1)

                    for neigh in range(min_y, max_y + 1):
                        neigh_node = previous_row_nodes[neigh]
                        adjacency_list += [(neigh_node.unique_id, current_node.unique_id),
                                           (current_node.unique_id, neigh_node.unique_id)]

            previous_row_nodes = deepcopy(current_row_nodes)
            current_row_nodes = []

        graph.add_edges_from(adjacency_list)

        all_nodes_features = []
        for node in nodes_list:
            all_nodes_features += [
                (
                    node.unique_id,
                    node.get_features_dict()
                )
            ]

        graph.add_nodes_from(all_nodes_features)

        return graph

    def generate(self):
        nx_graph = self.generate_nx_graph()
        return GraphState(nx_graph, nx_graph)

    def generate_multiple_graphs(self, quantity):
        return [self.generate() for _ in range(quantity)]



