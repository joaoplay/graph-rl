from copy import deepcopy

import networkx as nx

from environments.generator.fluid_network_node import FluidNetworkNode
from environments.generator.full_graph_generator import FullGraphGenerator
from environments.generator.graph_node import GraphNode
from environments.generator.utils import id_generator
from graphs.fluid_network_state import FluidNetworkState


class SingleVesselGraphGenerator:

    def __init__(self, size_x, size_y, interval_between_nodes, allow_void_actions=True) -> None:
        super().__init__()
        self.size_x = size_x
        self.size_y = size_y
        self.interval_between_nodes = interval_between_nodes
        self.allow_void_actions = allow_void_actions

    def get_node_type(self, x, y):
        if (x == 0 or x == 5) and y == 0:
            return 1
        elif (x == 0 or x == 5) and y == (self.size_y - 1):
            return 2
        else:
            return 0

    # Viscosity: 4 * 10-3
    # Diameter: 10
    # (pi * r^4 / 8) * 0.1mm
    
    # Raio converter para metros
    # 
    
    # 1mm
    
    # Factor de conversão 10e^12

    @staticmethod
    def get_pressure(node_type):
        if node_type == 1:
            # FIXME: 45
            return 10
        elif node_type == 2:
            # FIXME: 40
            return 5
        else:
            return 0

    @staticmethod
    def generate_row(row_length, nodes_list, start_node):
        single_vessel_adjacency_list = []
        previous_node = None
        for x in range(row_length):
            node = nodes_list[start_node + x]
            if previous_node:
                single_vessel_adjacency_list += [(previous_node.unique_id, node.unique_id)]
            previous_node = deepcopy(node)

        return single_vessel_adjacency_list

    def generate_nx_graph(self):
        graph = nx.DiGraph()

        id_gen = id_generator()

        nodes_list: list[GraphNode] = []
        for x in range(0, self.size_x, self.interval_between_nodes):
            for y in range(0, self.size_y, self.interval_between_nodes):
                node_type = self.get_node_type(x, y)
                pressure = self.get_pressure(node_type)
                current_node = FluidNetworkNode(unique_id=next(id_gen), coordinates=(x, y), z=1,  pressure=pressure,
                                                node_type=node_type)
                nodes_list += [current_node]

        row_length = int(self.size_x / self.interval_between_nodes)

        graph.add_edges_from(self.generate_row(row_length, nodes_list, 0))
        graph.add_edges_from(self.generate_row(row_length, nodes_list, 50))

        all_nodes_features = []
        for node in nodes_list:
            all_nodes_features += [
                (node.unique_id, node.get_features_dict())
            ]

        graph.add_nodes_from(all_nodes_features)

        return graph

    def _generate_neighbourhood_nx_graph(self):
        full_graph_generator = FullGraphGenerator(self.size_x, self.size_y, self.interval_between_nodes)
        nx_graph = full_graph_generator.generate_nx_graph()
        return nx_graph

    def generate(self):
        nx_graph = self.generate_nx_graph()
        neighbour_graph = self._generate_neighbourhood_nx_graph()
        return FluidNetworkState(nx_graph, neighbour_graph, allow_void_actions=self.allow_void_actions)

    def generate_multiple_graphs(self, quantity):
        return [self.generate() for _ in range(quantity)]
