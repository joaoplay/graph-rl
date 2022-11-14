import networkx as nx
import yaml

from environments.generator.fluid_network_node import FluidNetworkNode
from environments.generator.full_graph_generator import FullGraphGenerator
from environments.generator.graph_node import GraphNode
from environments.generator.utils import id_generator
from graphs.fluid_network_state import FluidNetworkState

INPUT_NODE_TYPE = 1
OUTPUT_NODE_TYPE = 2
NEUTRAL_NODE_TYPE = 0


class VasculatureNetworkFromFileGenerator:
    """
    This class is used to generate a vascular network.
    """

    def __init__(self, path):
        """
        This method is used to initialize the vascular network from file generator.
        :param path: The path to the file.
        """
        self.path = path
        self.env_size, self.input_nodes, \
        self.output_nodes, self.edges, \
        self.interval_between_nodes, self.static_pressures = self.read_network_from_file()

    def get_node_type(self, node_id):
        if node_id in self.input_nodes:
            return INPUT_NODE_TYPE
        elif node_id in self.output_nodes:
            return OUTPUT_NODE_TYPE
        else:
            return NEUTRAL_NODE_TYPE

    def get_pressure(self, node_id):
        return self.static_pressures.get(node_id, 0)

    def read_network_from_file(self):
        """
        This method is used to read the vascular network from file.
        :return: The vascular network.
        """
        with open(self.path) as file:
            net_structure = yaml.full_load(file)

            env_size = net_structure['size']
            input_nodes = net_structure['input_nodes']
            output_nodes = net_structure['output_nodes']
            edges = net_structure['edges']
            interval_between_nodes = net_structure['interval_between_nodes']
            static_pressures = net_structure['static_pressures']

            return env_size, input_nodes, output_nodes, edges, interval_between_nodes, static_pressures

    def generate_nx_graph(self):
        """
        This method is used to generate a vascular network.
        :return: The vascular network.
        """

        # Initialize an ID generator.
        id_gen = id_generator()

        # Initializer a list of nodes.
        nodes_list: list[GraphNode] = []
        for x in range(0, self.env_size['x'], self.interval_between_nodes):
            for y in range(0, self.env_size['y'], self.interval_between_nodes):
                node_id = next(id_gen)

                node_type = self.get_node_type(node_id)
                pressure = self.get_pressure(node_id)
                current_node = FluidNetworkNode(unique_id=node_id, coordinates=(x, y), z=1, pressure=pressure,
                                                node_type=node_type)
                nodes_list += [current_node]

        # Create empty graph
        graph = nx.DiGraph()

        print(self.edges[0][0])

        # Add edges
        # list_of_edges = [(edge[0], edge[1]) for edge in self.edges]
        graph.add_edges_from(self.edges)

        # Init node features
        all_nodes_features = []
        for node in nodes_list:
            all_nodes_features += [
                (node.unique_id, node.get_features_dict())
            ]
        graph.add_nodes_from(all_nodes_features)

        return graph

    def _generate_neighbourhood_nx_graph(self):
        full_graph_generator = FullGraphGenerator(self.env_size['x'], self.env_size['y'], self.interval_between_nodes)
        nx_graph = full_graph_generator.generate_nx_graph()

        return nx_graph

    def generate(self):
        nx_graph = self.generate_nx_graph()
        neighbour_graph = self._generate_neighbourhood_nx_graph()

        return FluidNetworkState(nx_graph, neighbour_graph)

    def generate_multiple_graphs(self, quantity):
        return [self.generate() for _ in range(quantity)]

