import networkx as nx
from matplotlib import pyplot as plt
from networkx import Graph

from graphs.graph_state import GraphState
from util import draw_nx_graph_with_coordinates


class FluidNetworkState(GraphState):
    """
    Represents a network with fluid mechanics. This network must have at least one input and output pressure.

    FIXME: Add details on the structure of graph features
    """

    def __init__(self, nx_graph: Graph, nx_neighbourhood_graph: Graph, allow_void_actions=True) -> None:
        super().__init__(nx_graph, nx_neighbourhood_graph, allow_void_actions)

    def get_node_features(self):
        """
        Get node features from a fluid network.
        :return:
        """
        nodes_x = nx.get_node_attributes(self.nx_graph, 'x')
        nodes_y = nx.get_node_attributes(self.nx_graph, 'y')
        coordinates = [[nodes_x[node_idx], nodes_y[node_idx]] for node_idx in range(self.nx_graph.number_of_nodes())]

        return coordinates

    def prepare_for_reward_evaluation(self):
        """
        Prepare a graph state before sending it to the fluid network
        :return:
        """
        nx_graph_copy = self.nx_graph.to_undirected()
        nx_graph_copy.remove_nodes_from(list(nx.isolates(nx_graph_copy)))

        # FIXME: Move it to a separated method
        nodes_data = nx_graph_copy.nodes(data=True)
        input_nodes = [x for x, y in nodes_data if y['node_type'] == 1]
        output_nodes = [x for x, y in nodes_data if y['node_type'] == 2]

        no_flow_nodes = set()
        for component in nx.connected_components(nx_graph_copy):
            in_node_found = False
            for in_node in input_nodes:
                if in_node in component:
                    in_node_found = True
                    break

            out_node_found = True
            for out_node in output_nodes:
                if out_node in component:
                    out_node_found = True
                    break

            if not in_node_found or not out_node_found:
                no_flow_nodes.update(component)

        nx_graph_copy.remove_nodes_from(no_flow_nodes)

        nx_graph_copy = nx.convert_node_labels_to_integers(nx_graph_copy, first_label=0, ordering='default',
                                                           label_attribute=None)

        inverse_edges = [(edge[1], edge[0]) for edge in nx_graph_copy.edges]

        edges_list = list(zip(*set(list(nx_graph_copy.edges) + inverse_edges)))

        # Sort node features by node index (ascending order)
        node_features_by_node = list(nx_graph_copy.nodes.data())
        node_features_by_node.sort()
        node_features = [list(node[1].values()) for node in node_features_by_node]

        if len(edges_list) > 0:
            edges_features = [[1] for _ in range(len(edges_list[0]))]
        else:
            return None

        return node_features, edges_list, edges_features
