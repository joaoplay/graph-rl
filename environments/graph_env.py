import time
from copy import deepcopy
from typing import Optional, List

import numpy as np
from matplotlib import pyplot as plt

from o2calculator.calculator import calculate_network_irrigation

from environments.stop_conditions import StopCondition
from graphs.edge_budget.base_edge_budget import BaseEdgeBudget
from graphs.edge_budget.fixed_edge_percentage_budget import FixedEdgePercentageBudget
from graphs.edge_budget.infinite_edge_budget import InfiniteEdgeBudget
from graphs.graph_state import GraphState
from settings import BASE_PATH
from util import draw_nx_graph_with_coordinates

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

    def __init__(self, stop_conditions: List[StopCondition], max_edges_percentage=None, stop_after_void_action: bool = False) -> None:
        super().__init__()

        # Batch of graphs
        self.graphs_list: Optional[list[GraphState]] = None
        # A 1D array with the same size as graphs_list length. It stores the reward values for each graph as long as the
        # finish conditions are met.
        self.edges_budget: Optional[BaseEdgeBudget] = None
        self.action_modes = DEFAULT_ACTION_MODES
        self.stop_conditions = stop_conditions
        self.steps_counter = 0
        self.stop_after_void_action = stop_after_void_action
        self.max_edges_percentage = max_edges_percentage
        self.action_type_statistics = []

        self.last_irrigation_map = None
        self.last_sources = None

        self.previous_irrigation_score = None

        self.rewards = None

    def step(self, actions):
        """
        This is the main method of this simulation environment. This is responsible for moving to the next time step,
        executing the corresponding action on each graph.

        :param actions: A list of actions (one of each graph in the graphs_list attribute)
        :return:
        """

        rewards = np.zeros(len(self.graphs_list))

        for graph_idx in range(len(self.graphs_list)):
            if self.edges_budget.is_exhausted(graph_idx):
                # The current graph budget is exhausted. Just ignore the current graph and move to the next one.
                continue

            if self.stop_after_void_action and actions[graph_idx] == -1:
                raise Exception("Invalid action found")
                # self.edges_budget.force_exhausting(graph_idx)
                # continue

            # Get the current graph state and its remaining edges budget
            current_graph = self.graphs_list[graph_idx]
            remaining_budget = self.edges_budget.get_remaining_budget(graph_idx)

            start_node = self.graphs_list[graph_idx].selected_start_node

            # Execute action and get the resulting graph (a deepcopy) and the insertion cost (in terms of edge budget)
            new_graph, edge_insertion_cost = self.execute_action(current_graph, actions[graph_idx], remaining_budget)

            # Log statistics
            if edge_insertion_cost == 1:
                self.action_type_statistics[graph_idx] += [0]
            elif edge_insertion_cost == -1:
                self.action_type_statistics[graph_idx] += [1]

            # Save the new graph
            self.graphs_list[graph_idx] = new_graph

            if self.current_action_mode == ACTION_MODE_SELECTING_END_NODE:
                node_added = edge_insertion_cost > 0
                rewards[graph_idx] = self.calculate_reward(graph_idx=graph_idx, node_added=node_added, start_node=start_node, end_node=actions[graph_idx])
            else:
                rewards[graph_idx] = 0

            # Update edges budget
            self.edges_budget.increment_used_budget(graph_idx, edge_insertion_cost)

            if self.current_action_mode == ACTION_MODE_SELECTING_END_NODE \
               and self.graphs_list[graph_idx].allowed_actions_not_found:
                # A new edge was added and no valid start nodes are available. The current graph reached a dead end, and therefore
                # it's time to end the generation process.
                self.edges_budget.force_exhausting(graph_idx)

        self.rewards = rewards
        self.steps_counter += 1

    @property
    def current_action_mode(self):
        """
        Get the current action mode. It can be either ACTION_MODE_SELECTING_START_NODE or ACTION_MODE_SELECTING_END_NODE
        :return: Either ACTION_MODE_SELECTING_START_NODE or ACTION_MODE_SELECTING_END_NODE
        """
        return self.steps_counter % len(self.action_modes)

    def execute_action(self, graph: GraphState, action, remaining_budget):
        """
        Execute an action on a given graph. This is applied to start node and end node selection.
        :param graph: A Graph state object
        :param action: A Node ID to be selected
        :param remaining_budget: The current remaining edge budget of the graph
        :return:
        """
        if not graph.start_node_is_selected:
            # A start node is not selected

            # if not graph.allowed_actions_not_found:
            # Ensure that the start node selection is feasible.

            graph.select_start_node(action)

            # Update forbidden actions
            graph.populate_forbidden_actions(remaining_budget)

            return graph, 0
        else:
            # A start node is already selected. It's time to select the end node and create a new edge.
            edge_insertion_cost = 0

            #if not graph.allowed_actions_not_found:
            # Ensure that the start node selection is feasible.

            graph, edge_insertion_cost = graph.add_or_remove_edge(graph.selected_start_node, action)

            # Invalidate the currently selected start node. A new start node will be selected in the next simulation step.
            graph.invalidate_selected_start_node()

            # Update forbidden actions
            graph.populate_forbidden_actions(remaining_budget - edge_insertion_cost)

            return graph, edge_insertion_cost

    def init(self, graphs_list):
        """
        Initialize a new simulation environment that includes the graphs in the graphs_list parameter.
        :param graphs_list: A list of GraphState objects.
        :return:
        """
        self.graphs_list = graphs_list

        if self.max_edges_percentage is None:
            self.edges_budget = InfiniteEdgeBudget(self.graphs_list)
        else:
            self.edges_budget = FixedEdgePercentageBudget(self.graphs_list, fixed_edge_percentage=self.max_edges_percentage)

        # Init simulation statistics array
        self.action_type_statistics = [[] for _ in range(len(graphs_list))]

        self.steps_counter = 0

        self.last_irrigation_map = None
        self.last_sources = None
        self.previous_irrigation_score = [self.calculate_reward(graph_idx=graph_idx) for graph_idx in range(len(graphs_list))]

        self.rewards = None

        for graph_idx in range(len(self.graphs_list)):
            graph = self.graphs_list[graph_idx]
            graph.invalidate_selected_start_node()
            graph.populate_forbidden_actions(self.edges_budget.get_remaining_budget(graph_idx))

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
    def non_exhausted_graphs(self):
        """
        All graphs whose edges budget was not exceeded.
        :return: A list of Graph State objects.
        """
        all_non_exhausted_ids = self.edges_budget.all_non_exhausted
        return [self.graphs_list[idx] for idx in all_non_exhausted_ids]

    @property
    def non_exhausted_graph_ids(self):
        """
        Node IDs of all non-exhausted graphs
        :return:
        """
        return self.edges_budget.all_non_exhausted

    @property
    def exhausted_graph_ids(self):
        """
            Node IDs of all exhausted graphs
            :return:
        """
        return self.edges_budget.all_exhausted

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

    def is_terminal(self):
        """
        Check all stop conditions
        :return:
        """
        return any([sc.is_satisfied(self) for sc in self.stop_conditions])

    def calculate_reward_all_graphs(self):
        rewards = np.zeros(len(self.graphs_list), dtype=np.float)
        for graph_idx in range(len(self.graphs_list)):
            reward = self.calculate_reward(graph_idx)
            if abs(reward) < REWARD_EPS:
                reward = 0

            # Save reward
            rewards[graph_idx] = reward

        return rewards

    def calculate_reward(self, graph_idx, node_added=True, start_node=None, end_node=None):
        graph = self.graphs_list[graph_idx]

        prepared_data = graph.prepare_for_reward_evaluation(node_added=node_added, start_node=start_node, end_node=end_node)

        if not prepared_data:
            return 0

        """fig, ax = plt.subplots()
        draw_nx_graph_with_coordinates(prepared_data[3], ax)
        fig.savefig(f'{BASE_PATH}/test_images/graph-sim-{self.steps_counter + 1}.png')

        fig, ax = plt.subplots()
        draw_nx_graph_with_coordinates(graph.nx_graph, ax)
        fig.savefig(f'{BASE_PATH}/test_images/graph-{self.steps_counter + 1}.png')"""

        irrigation, sources = calculate_network_irrigation(prepared_data[0], prepared_data[1], prepared_data[2], [10, 10], [0.1, 0.1])

        sections_x = np.array_split(irrigation, 20, axis=0)
        sections_y = np.array_split(irrigation, 20, axis=1)

        irrigation_score_x = sum([np.mean(section) for section in sections_x])
        irrigation_score_y = sum([np.mean(section) for section in sections_y])

        irrigation_score = (irrigation_score_x + irrigation_score_y) / 2.0

        if not self.previous_irrigation_score:
            return irrigation_score

        irrigation_improvement = irrigation_score - self.previous_irrigation_score[graph_idx]

        # Update irrigation map
        self.last_irrigation_map = irrigation
        self.last_sources = sources

        """fig, ax = plt.subplots()
        ax.imshow(np.flipud(irrigation), cmap='hot', interpolation='nearest')
        fig.savefig(f'{BASE_PATH}/test_images/heatmap-{self.steps_counter + 1}.png')"""

        return irrigation_improvement
