from copy import deepcopy
from typing import Optional

import numpy as np
import skimage.measure

from o2calculator.calculator import calculate_network_irrigation

from graphs.graph_state import GraphState

REWARD_EPS = 1e-4

ACTION_MODE_SELECTING_START_NODE = 0
ACTION_MODE_SELECTING_END_NODE = 1
DEFAULT_ACTION_MODES = (ACTION_MODE_SELECTING_START_NODE, ACTION_MODE_SELECTING_END_NODE)


class GraphEnv:
    """
    This class represents a simulation environment composed of multiple graphs (stored in the graph_list attribute).

    An environment a step() method responsible for moving to the next time step. It consists of applying a set of actions
    on each graph.

    Reward Assignment: The model only allows EPISODIC reward. Intermediate rewards are not calculated and will be implemented included later.

    Edges Budget: Graph construction can be constrained by a maximum number of edges in the whole graph. For that purpose,
                  an edges budget is defined beforehand and updated as new edges are added up.

    Mandatory overriding:

        - is_terminal(): Returns a boolean value indicating whether a simulation is over. Please remember
                         that a graph environment deals with a batch of graphs. This method operates over all graphs at
                         the same time

        - calculate_reward(graph: GraphState): Returns the reward value for a given graph (GraphState instance)

    Simulation Stop Conditions: The simulation stops whenever all stop condition are satisfied.


    """

    def __init__(self, max_steps, irrigation_goal, inject_irrigation,
                 irrigation_compression, irrigation_grid_dim, irrigation_grid_cell_size, irrigation_percentage_goal,
                 exclude_isolated_from_start_nodes=False, use_irrigation_improvement=False) -> None:
        super().__init__()

        # Batch of graphs
        self.graphs_list: Optional[list[GraphState]] = None
        self.action_modes = DEFAULT_ACTION_MODES
        self.max_steps = max_steps
        self.irrigation_goal = irrigation_goal
        self.inject_irrigation = inject_irrigation
        self.irrigation_compression = irrigation_compression
        self.irrigation_grid_dim = np.array(irrigation_grid_dim)
        self.irrigation_grid_cell_size = np.array(irrigation_grid_cell_size)
        self.irrigation_percentage_goal = irrigation_percentage_goal
        self.use_irrigation_improvement = use_irrigation_improvement

        self.steps_counter = 0
        self.action_type_statistics = []

        self.done = None
        self.solved = None

        # FIXME: Change this to a class that contains historic information
        self.last_irrigation_map = None
        self.last_sources = None
        self.last_pressures = None
        self.last_irrigation_graph = None
        self.last_edge_sources = None
        self.last_edges_list = None

        self.previous_irrigation_score = None

        self.start_node_selection_statistics = None
        self.end_node_selection_statistics = None
        self.repeated_actions = 0

        self.episode_retrospective = []

        self.exclude_isolated_from_start_nodes = exclude_isolated_from_start_nodes

    @property
    def compressed_irrigation_matrix_size(self):
        matrix_dim = np.divide((self.irrigation_grid_dim - 1), self.irrigation_grid_cell_size).astype(int) + 1
        dummy_matrix = np.zeros(matrix_dim)
        compressed_matrix = skimage.measure.block_reduce(dummy_matrix,
                                                         (self.irrigation_compression, self.irrigation_compression),
                                                         np.max)
        return np.prod(compressed_matrix.shape)

    def step(self, actions):
        """
        This is the main method of this simulation environment. This is responsible for moving to the next time step,
        executing the corresponding action on each graph.

        :param actions: A list of actions (one of each graph in the graphs_list attribute)
        :return:
        """

        rewards = np.zeros(len(self.graphs_list))
        for graph_idx in range(len(self.graphs_list)):
            if self.done[graph_idx]:
                continue

            if actions[graph_idx] == -1:
                raise Exception("Invalid action found")

            # Get the current graph state and its remaining edges budget
            current_graph = self.graphs_list[graph_idx]

            start_node = self.graphs_list[graph_idx].selected_start_node

            # Execute action and get the resulting graph (a deepcopy) and the insertion cost (in terms of edge budget)
            new_graph, edge_insertion_cost = self.execute_action(current_graph, actions[graph_idx])

            # Log statistics
            if edge_insertion_cost == 1:
                self.action_type_statistics[graph_idx] += [0]
            elif edge_insertion_cost == -1:
                self.action_type_statistics[graph_idx] += [1]

            # Save the new graph
            self.graphs_list[graph_idx] = new_graph

            if self.current_action_mode == ACTION_MODE_SELECTING_START_NODE:
                if current_graph.nx_graph.degree[actions[graph_idx]] > 2:
                    rewards[graph_idx] = -1

            if self.current_action_mode == ACTION_MODE_SELECTING_END_NODE:
                node_added = edge_insertion_cost > 0
                # Calculate irrigation in order to check if the episode is over
                irrigation_improvement = self.calculate_reward(graph_idx=graph_idx, node_added=node_added,
                                                               start_node=start_node, end_node=actions[graph_idx])

                rewards[graph_idx] = -1.0
                if self.use_irrigation_improvement:
                    rewards[graph_idx] += irrigation_improvement

            # FIXME: The irrigation map only support 1 graph. Adapt it for multi graph
            if self.irrigation_goal_achieved() or self.max_steps_achieved() or self.irrigation_percentage_goal_achieved():
                self.done[graph_idx] = True

                if self.irrigation_goal_achieved() or self.irrigation_percentage_goal_achieved():
                    self.solved[graph_idx] = True

            if new_graph.allowed_actions_not_found:
                print("Allowed actions not found")
                self.done[graph_idx] = True
                self.solved[graph_idx] = False
                exit(1)

        self.steps_counter += 1

        return self.current_graph_representation_copy, rewards, deepcopy(self.done), deepcopy(self.solved)

    @staticmethod
    def _get_random_action(graph: GraphState):
        # Get all possible start nodes
        valid_start_nodes = graph.allowed_actions
        if len(valid_start_nodes) == 0:
            # No start nodes available
            return -1, -1

        # Choose a start node
        start_node = np.random.choice(list(valid_start_nodes))

        # Get all possible end nodes
        valid_end_nodes = graph.get_valid_end_nodes(start_node=start_node)

        # print(f"Start Node {start_node} | Invalid End Nodes: {graph.get_invalid_end_nodes(start_node=start_node)}")

        end_node = np.random.choice(list(valid_end_nodes))

        return start_node, end_node

    def get_random_actions(self):
        selected_start_nodes = []
        selected_end_nodes = []

        for graph in self.graphs_list:
            start_node, end_node = self._get_random_action(graph)

            selected_start_nodes += [start_node]
            selected_end_nodes += [end_node]

        return selected_start_nodes, selected_end_nodes

    def max_steps_achieved(self):
        return self.steps_counter >= self.max_steps - 1

    def irrigation_goal_achieved(self):
        if self.last_irrigation_map is None:
            return False

        return np.all((self.last_irrigation_map > self.irrigation_goal))

    def irrigation_percentage_goal_achieved(self):
        return self.previous_irrigation_score[0] >= self.irrigation_percentage_goal

    @property
    def current_action_mode(self):
        """
        Get the current action mode. It can be either ACTION_MODE_SELECTING_START_NODE or ACTION_MODE_SELECTING_END_NODE
        :return: Either ACTION_MODE_SELECTING_START_NODE or ACTION_MODE_SELECTING_END_NODE
        """
        return self.steps_counter % len(self.action_modes)

    def execute_action(self, graph: GraphState, action):
        """
        Execute an action on a given graph. This is applied to start node and end node selection.
        :param graph: A Graph state object
        :param action: A Node ID to be selected
        :return:
        """
        if not graph.start_node_is_selected:
            # A start node is not selected

            # if not graph.allowed_actions_not_found:
            # Ensure that the start node selection is feasible.

            graph.select_start_node(action)

            self.start_node_selection_statistics[action] += 1

            # Update forbidden actions
            graph.populate_forbidden_actions()

            return graph, 0
        else:
            # A start node is already selected. It's time to select the end node and create a new edge.
            edge_insertion_cost = 0

            # if not graph.allowed_actions_not_found:
            # Ensure that the start node selection is feasible.

            previous_selected_start_node = graph.selected_start_node

            neighbours = list(graph.nx_neighbourhood_graph.neighbors(graph.selected_start_node))

            end_node = neighbours[action]
            graph, edge_insertion_cost = graph.add_or_remove_edge(graph.selected_start_node, end_node)

            graph.previous_selected_start_node = previous_selected_start_node

            self.end_node_selection_statistics[action] += 1

            # Invalidate the currently selected start node. A new start node will be selected in the next simulation step.
            graph.invalidate_selected_start_node()

            # Update forbidden actions
            graph.populate_forbidden_actions()

            return graph, edge_insertion_cost

    def init(self, graphs_list):
        """
        Initialize a new simulation environment that includes the graphs in the graphs_list parameter.
        :param graphs_list: A list of GraphState objects.
        :return:
        """
        self.graphs_list = graphs_list

        # Exclude isolated nodes from start nodes when required
        if self.exclude_isolated_from_start_nodes:
            for graph in self.graphs_list:
                graph.exclude_isolated_from_start_nodes = True

        # Init simulation statistics array
        self.action_type_statistics = [[] for _ in range(len(graphs_list))]
        self.done = [False for _ in range(len(graphs_list))]
        self.solved = [False for _ in range(len(graphs_list))]

        self.steps_counter = 0

        self.last_irrigation_map = None
        self.last_sources = None
        self.last_pressures = None
        self.last_irrigation_graph = None
        self.last_edge_sources = None
        self.last_edges_list = None
        self.previous_irrigation_score = None

        self.previous_irrigation_score = [self.calculate_reward(graph_idx=graph_idx) for graph_idx in
                                          range(len(graphs_list))]

        self.start_node_selection_statistics = {node: 0 for node in self.graphs_list[0].nx_graph.nodes}
        self.end_node_selection_statistics = {node: 0 for node in self.graphs_list[0].nx_graph.nodes}

        self.repeated_actions = 0

        for graph_idx in range(len(self.graphs_list)):
            graph = self.graphs_list[graph_idx]
            graph.invalidate_selected_start_node()
            graph.populate_forbidden_actions()

    @property
    def current_state(self):
        """
        Get the current state of all graphs in the simulation environment.
        :return: A zip that contains (graph_states, selected_start_nodes, forbidden_actions).

        Example:

            graph_state[0]-> Graph State of Graph 0
            selected_start_nodes[0] -> Selected Start Node of graph 0 (None whenever a start node is not selected)
            forbidden_actions[0] -> Forbidden actions (a list of Node IDs) of graph 0
        """
        start_nodes = [g.selected_start_node for g in self.graphs_list][:]
        forbidden_actions = [g.forbidden_actions for g in self.graphs_list][:]
        return zip(self.graphs_list, start_nodes, forbidden_actions)

    @property
    def current_graph_representation_copy(self):
        graph_representations = GraphState.convert_all_to_representation(self.current_action_mode, self.graphs_list)

        if self.inject_irrigation:
            compressed_irrigation = self.last_irrigation_map
            if self.irrigation_compression:
                compressed_irrigation = skimage.measure.block_reduce(self.last_irrigation_map, (
                self.irrigation_compression, self.irrigation_compression), np.max)

            # Inject irrigation matrix when required
            irrigated_cells = np.where(compressed_irrigation > self.irrigation_goal, 1, 0).reshape(1, -1).astype(
                np.float32)
            graph_representations = np.concatenate((graph_representations, irrigated_cells), axis=1)

        return graph_representations

    def clone_current_state(self, graph_indexes=None):
        """
        Clone a set of graphs. All graphs are returned whenever the graph_indexes parameter is None.
        :param graph_indexes: A list of Graph IDs
        :return: A list of Graph State objects.
        """
        if not graph_indexes:
            graph_indexes = list(range(len(graph_indexes)))

        return [(deepcopy(self.graphs_list[i]), deepcopy(self.graphs_list[i].selected_start_node),
                 deepcopy(self.graphs_list[i].forbidden_actions))
                for i in graph_indexes]

    def calculate_reward_all_graphs(self):
        rewards = np.zeros(len(self.graphs_list), dtype=np.float)
        for graph_idx in range(len(self.graphs_list)):
            reward = self.calculate_reward(graph_idx)
            if abs(reward) < REWARD_EPS:
                reward = 0

            # Save reward
            rewards[graph_idx] = reward

        return rewards

    def calculate_reward(self, graph_idx, node_added=True, start_node=None, end_node=None, reward_multi=10):
        graph = self.graphs_list[graph_idx]

        prepared_data = graph.prepare_for_reward_evaluation(node_added=node_added, start_node=start_node,
                                                            end_node=end_node)

        irrigation_improvement = 0
        if prepared_data is not None:
            if prepared_data == -1:  # No irrigation
                self.last_irrigation_map = None
                self.previous_irrigation_score[graph_idx] = 0
            elif prepared_data != -1:
                irrigation, sources, pressures, edges_source, edges_list = calculate_network_irrigation(
                    prepared_data[0], prepared_data[1],
                    prepared_data[2], self.irrigation_grid_dim, self.irrigation_grid_cell_size)

                # sections_x = np.array_split(irrigation, 20, axis=0)
                # sections_y = np.array_split(irrigation, 20, axis=1)

                """def percentage_irrigated(sections):
                    irrigated = []
                    for s in sections:
                        irrigated += [np.count_nonzero(s[s > self.irrigation_goal]) / s.size]
                    return np.array(irrigated)

                irrigated_x = percentage_irrigated(sections_x)
                irrigated_y = percentage_irrigated(sections_y)

                mean_irrigated_x = np.mean(irrigated_x)
                mean_irrigated_y = np.mean(irrigated_y)

                irrigation_score = (mean_irrigated_x + mean_irrigated_y) / 2.0"""

                # irrigation_score = np.count_nonzero(irrigation[irrigation > self.irrigation_goal]) / irrigation.size

                over_irrigation_value = irrigation - self.irrigation_goal
                over_irrigation_value[over_irrigation_value < 0] = -1
                over_irrigation_value[over_irrigation_value != -1] = np.power(0.8, over_irrigation_value[over_irrigation_value != -1])

                irrigation_score = np.mean(over_irrigation_value)

                if self.previous_irrigation_score is not None:
                    irrigation_improvement = irrigation_score - self.previous_irrigation_score[graph_idx]

                if self.previous_irrigation_score:
                    self.previous_irrigation_score[graph_idx] = irrigation_score

                self.last_irrigation_map = irrigation
                self.last_irrigation_graph = prepared_data[3]
                self.last_sources = sources
                self.last_edge_sources = edges_source
                self.last_pressures = pressures
                self.last_edges_list = edges_list

        return irrigation_improvement

    """def calculate_reward(self, graph_idx, node_added=True, start_node=None, end_node=None, reward_multi=10):
        graph = self.graphs_list[graph_idx]

        prepared_data = graph.prepare_for_reward_evaluation(node_added=node_added, start_node=start_node,
                                                            end_node=end_node)

        irrigation_score = 0
        irrigation = None
        sources = None
        if not prepared_data:
            return 0
        elif prepared_data != -1:
            irrigation, sources = calculate_network_irrigation(prepared_data[0], prepared_data[1], prepared_data[2],
                                                               [5, 5], [0.1, 0.1])

            sections_x = np.array_split(irrigation, 20, axis=0)
            sections_y = np.array_split(irrigation, 20, axis=1)

            mean_over_x = [np.mean(section) for section in sections_x]
            mean_over_y = [np.mean(section) for section in sections_y]

            irrigation_score_x = sum(mean_over_x)
            irrigation_score_y = sum(mean_over_y)

            irrigation_score = (irrigation_score_x + irrigation_score_y) / 2.0

        if not self.previous_irrigation_score:
            return irrigation_score * reward_multi

        # print(f"Previous Irrigation Score: {self.previous_irrigation_score[graph_idx] * reward_multi}")

        irrigation_improvement = irrigation_score - self.previous_irrigation_score[graph_idx]

        # Update irrigation map
        self.previous_irrigation_score[graph_idx] = irrigation_score
        self.last_irrigation_map = irrigation
        self.last_sources = sources

        # print(f"Step: {self.steps_counter + 1} | Reward: {irrigation_improvement * reward_multi}")

        return irrigation_improvement * reward_multi"""
