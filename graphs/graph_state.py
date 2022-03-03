from copy import deepcopy

import networkx as nx
import numpy as np

BUDGET_EPS = 1e-5

DEFAULT_EDGE_INSERTION_COST = 1


class GraphState:
    """
    A graph state represents a single graph in a graph environments. Remember that an environments (see GraphEnv) is
    composed of many graphs, each represented by a GraphState object. Therefore, a GraphState holds every feature
    and conditions of a given graph in the simulation environments. Apart from the structural characteristics, a GraphState
    keeps track on important attributes regarding the graph generation process. For instance, every time a node is selected,
    the selected_start_node attribute stores the corresponding node ID.

    Note that a Graph state may contain isolated nodes.

    Very important: A graph state must contain nodes with consecutive IDs, starting ate the 0 index. Otherwise, the whole algorithm fails.

    For example:

        Valid ID order: [0, 1, 2, 3]
        Invalid ID order: [1, 2, 3, 4] or [0, 4, 6]

    Tip: Network X provides a very useful method to reset IDs, starting from 0 index and following a consecutive step.

         ```nx.convert_node_labels_to_integers(nx_graph, first_label=0, ordering='default', label_attribute=None)```
    """

    def __init__(self, nx_graph, nx_neighbourhood_graph, allow_void_actions=True) -> None:
        """
        :param nx_graph: The NetworkX Graph representing the current graph structure.
        :param nx_neighbourhood_graph: A NetworkX Graph which represent the valid neighborhood of each node.
        """

        super().__init__()

        # Total number of nodes. Note that isolated nodes are counted in.
        self.num_nodes = nx_graph.number_of_nodes()
        # Store all node IDs
        self.node_labels = np.arange(self.num_nodes)
        # The NetworkX Graph representing the current graph structure
        self.nx_graph = nx_graph
        # A NetworkX Graph which represent the valid neighborhood of each node.
        self.nx_neighbourhood_graph = nx_neighbourhood_graph

        # A set with all nodes in the graph. This is useful to intercept sets of nodes.
        self.all_nodes_set = set(self.node_labels)
        # Keeps track on the degree of each node in the graph
        self.node_degrees = np.array([deg for (node, deg) in sorted(nx_graph.degree(), key=lambda deg_pair: deg_pair[0])])

        # Decompose start and end node of every edge
        x, y = zip(*nx_graph.edges())
        # Store the number of edges
        self.num_edges = len(x)

        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = np.ravel(self.edge_pairs)

        # Keep track on the currently selected node
        self.selected_start_node = None
        # This attribute is very important. At the end of a given simulation step, every forbidden action (for the state
        # after applying a given action) is stored and essential to discard every invalid action
        self.forbidden_actions = None

        self.allow_void_actions = allow_void_actions

    def invalidate_selected_start_node(self):
        """
        Reset the currently selected start node.
        :return:
        """
        self.selected_start_node = None

    @property
    def start_node_is_selected(self):
        """
        Check whether a start node is selected.
        :return: Returns True whenever a start node is selected
        """
        return self.selected_start_node is not None

    def select_start_node(self, start_node_id):
        """
        Select a start node. This method is called whenever a start node selection action is executed.
        :param start_node_id: Node ID
        :return:
        """
        self.selected_start_node = start_node_id

    def add_or_remove_edge(self, start_node, end_node):
        if self.nx_graph.has_edge(start_node, end_node):
            self.nx_graph.remove_edge(start_node, end_node)
            edge_cost = -DEFAULT_EDGE_INSERTION_COST
        elif self.nx_graph.has_edge(end_node, start_node):
            self.nx_graph.remove_edge(end_node, start_node)
            edge_cost = -DEFAULT_EDGE_INSERTION_COST
        else:
            self.nx_graph.add_edge(start_node, end_node)
            edge_cost = DEFAULT_EDGE_INSERTION_COST

        graph_state = self.__class__(self.nx_graph, self.nx_neighbourhood_graph)

        return graph_state, edge_cost

    def add_edge(self, start_node, end_node):
        """
        Create an edge between to nodes in the graph
        :param start_node:
        :param end_node:
        :return: A new object (deep copy of the nx_graph) is returned
        """
        nx_graph_clone = self.nx_graph
        nx_graph_clone.add_edge(start_node, end_node)

        graph_state = self.__class__(nx_graph_clone, self.nx_neighbourhood_graph)
        return graph_state, DEFAULT_EDGE_INSERTION_COST

    def remove_edge(self, start_node, end_node):
        nx_graph_clone = self.nx_graph
        nx_graph_clone.remove_edge(start_node, end_node)

        graph_state = self.__class__(nx_graph_clone, self.nx_neighbourhood_graph)
        return graph_state, -DEFAULT_EDGE_INSERTION_COST

    def get_node_features(self):
        raise NotImplementedError()

    def prepare_for_reward_evaluation(self, node_added=True, start_node=None, end_node=None):
        raise NotImplementedError()

    def get_valid_end_nodes(self, start_node=None):
        """
        Get all valid end nodes
        :param start_node:
        :return:
        """
        return self.all_nodes_set - self.get_invalid_end_nodes(start_node=start_node)

    def get_invalid_start_nodes(self):
        """
        Get all  invalid start nodes
        :return:
        """
        # Identify all isolated nodes. Nodes with zero degree
        isolated_nodes = set(nx.isolates(self.nx_graph))
        # Identify nodes with no edges available. FIXME: Is it correct? Probably we should check the neighborhood graph instead.
        # nodes_with_no_edges_available = set([node_id for node_id in self.node_labels
        #                                   if self.node_degrees[node_id] == (self.num_nodes - 1)])

        """invalid_nodes = set()
        if not self.allow_void_actions:
            # Pick all nodes that are neither isolated nor full of edges
            remaining_nodes = self.all_nodes_set - isolated_nodes - nodes_with_no_edges_available
            invalid_nodes = set([node for node in remaining_nodes if len(self.get_invalid_end_nodes(start_node=node)) == self.num_nodes])"""

        return set()  # isolated_nodes

    def get_invalid_end_nodes(self, start_node=None):
        # Use the start_node passed as parameter whenever defined. Otherwise, use the selected start node
        # We support a specific start node (apart from the one specified in selected_start_node) to mock a specific move
        start_node = start_node if start_node is not None else self.selected_start_node

        invalid_end_nodes = set()
        invalid_end_nodes.add(start_node)

        # existing_edges = self.edge_pairs.reshape(-1, 2)

        # Exclude all nodes that already have an edge FROM the selected node
        # existing_left = existing_edges[existing_edges[:, 0] == start_node]
        # invalid_end_nodes.update(np.ravel(existing_left[:, 1]))

        # Exclude all nodes that already have an edge TO the selected node
        # existing_right = existing_edges[existing_edges[:, 1] == start_node]
        # invalid_end_nodes.update(np.ravel(existing_right[:, 0]))

        select_node_neighbors = set(self.nx_neighbourhood_graph.neighbors(start_node))
        all_nodes = set(self.nx_neighbourhood_graph.nodes)
        non_neighbor_nodes = all_nodes - select_node_neighbors

        invalid_end_nodes.update(non_neighbor_nodes)

        return invalid_end_nodes

    def populate_forbidden_actions(self, budget=None):
        if budget is not None:
            if budget < BUDGET_EPS:
                self.forbidden_actions = self.all_nodes_set
                return

        if self.selected_start_node is None:
            # A start node is not selected
            self.forbidden_actions = self.get_invalid_start_nodes()
        else:
            self.forbidden_actions = self.get_invalid_end_nodes()

    @staticmethod
    def generate_nodes_attributes_with_selected_node(graph, selected_node_id):
        all_nodes = {node: {"is_selected": 0.0} for node in graph.nx_graph.nodes()}

        if selected_node_id is not None:
            all_nodes[selected_node_id]['is_selected'] = 1.0

        return all_nodes

    @staticmethod
    def get_non_isolated_nodes(graphs):
        n_nodes = 0

        isolated_nodes_indexes = []
        for graph in graphs:
            isolated_nodes_indexes += [n_nodes + node_idx for node_idx in list(nx.isolates(graph.nx_graph))]
            n_nodes += graph.num_nodes

        non_isolated = np.ones((len(graphs), graphs[0].num_nodes))
        non_isolated.put(isolated_nodes_indexes, 0)

        return non_isolated

    @property
    def allowed_actions_not_found(self):
        return len(self.forbidden_actions) == len(self.all_nodes_set)

    @property
    def allowed_actions(self):
        return self.all_nodes_set - self.forbidden_actions



