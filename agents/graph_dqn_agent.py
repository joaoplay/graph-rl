import collections
import itertools
import logging
import os
import time
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.autograd import Variable
from tqdm import tqdm

from agents.base_agent import BaseAgent
from agents.replay_memory.multi_action_experience_buffer import MultiActionModeExperienceBuffer
from agents.util.sample_tracker import BatchSampler
from environments.graph_env import DEFAULT_ACTION_MODES, ACTION_MODE_SELECTING_START_NODE, \
    ACTION_MODE_SELECTING_END_NODE
from environments.graph_env import GraphEnv
from graphs.graph_state import GraphState
import neptune_logging
from models.multi_action_mode_dqn import MultiActionModeDQN
from settings import USE_CUDA, BASE_PATH, NEPTUNE_INSTANCE
from util import draw_nx_graph_with_coordinates

log = logging.getLogger(__name__)


class GraphDQNAgent(BaseAgent):
    """
    GraphDQNAgent is responsible for interacting with a batch of graphs, adding and removing edges between two nodes. The
    agent's policy is learned from a Q-Learning process. Specifically, q-values are predicted by neural networks trained
    with data from an experience replay.

    At each time step, the agent performs one of the following actions:
        - Select a start node to be the new edge
        - Select an end nodes from the previous node neighborhood (currently a Moore neighborhood)

    Each action is governed by a distinct neural networks defined inside MultiActionModeDQN class. Analogously, the
    experience buffer is independent for each type of action.

    The core methods of this class are run_simulation() and train(). They are responsible for starting a fresh simulation
    and training Q-network, respectively.
    """

    def __init__(self, environment: GraphEnv, start_node_selection_dqn_params: Dict,
                 end_node_selection_dqn_params: Dict, batch_size: int = 50, warm_up: int = 4,
                 learning_rate: float = 0.0001, eps_start: float = 1, eps_step_denominator: float = 2,
                 eps_end: float = 0.1, validation_interval: int = 1000,
                 target_network_copy_interval: int = 50, action_modes: tuple[int] = DEFAULT_ACTION_MODES) -> None:
        """
        :param environment: A GraphEnv environments
        :param batch_size: The number of graphs given to the agent in a single simulation. The agent plays every GraphEnv
                           "at the same time"
        :param warm_up: Execute N simulations before starting the training process. These simulations do not affect training
                        and are useful for sanity check purpose
        :param learning_rate: FIXME
        """
        super().__init__(environment)

        self.action_modes = action_modes
        self.experience_buffers = MultiActionModeExperienceBuffer(action_modes)
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step_denominator = eps_step_denominator
        self.eps = None
        self.eps_step = None
        self.q_networks = MultiActionModeDQN(action_modes=self.action_modes,
                                             embedding_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: start_node_selection_dqn_params[
                                                     'embedding_dim'],
                                                 ACTION_MODE_SELECTING_END_NODE: end_node_selection_dqn_params[
                                                     'embedding_dim'],
                                             },
                                             hidden_output_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: start_node_selection_dqn_params[
                                                     'hidden_output_dim'],
                                                 ACTION_MODE_SELECTING_END_NODE: start_node_selection_dqn_params[
                                                     'hidden_output_dim'],
                                             },
                                             num_node_features={
                                                 ACTION_MODE_SELECTING_START_NODE: start_node_selection_dqn_params[
                                                     'num_node_features'],
                                                 ACTION_MODE_SELECTING_END_NODE: start_node_selection_dqn_params[
                                                     'num_node_features'],
                                             },
                                             action_output_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: 25,
                                                 ACTION_MODE_SELECTING_END_NODE: 25,
                                             })
        self.target_q_networks = MultiActionModeDQN(action_modes=self.action_modes,
                                                    embedding_dim={
                                                        ACTION_MODE_SELECTING_START_NODE:
                                                            start_node_selection_dqn_params['embedding_dim'],
                                                        ACTION_MODE_SELECTING_END_NODE: end_node_selection_dqn_params[
                                                            'embedding_dim'],
                                                    },
                                                    hidden_output_dim={
                                                        ACTION_MODE_SELECTING_START_NODE:
                                                            start_node_selection_dqn_params['hidden_output_dim'],
                                                        ACTION_MODE_SELECTING_END_NODE: start_node_selection_dqn_params[
                                                            'hidden_output_dim'],
                                                    },
                                                    num_node_features={
                                                        ACTION_MODE_SELECTING_START_NODE:
                                                            start_node_selection_dqn_params['num_node_features'],
                                                        ACTION_MODE_SELECTING_END_NODE: start_node_selection_dqn_params[
                                                            'num_node_features'],
                                                    },
                                                    action_output_dim={
                                                        ACTION_MODE_SELECTING_START_NODE: 25,
                                                        ACTION_MODE_SELECTING_END_NODE: 25,
                                                    })

        if USE_CUDA == 1:
            self.q_networks = self.q_networks.cuda()
            self.target_q_networks = self.target_q_networks.cuda()

        self.target_network_copy_interval = target_network_copy_interval
        self.validation_interval = validation_interval
        self.current_action_mode = None
        self.current_training_step = 0
        # This variable is defined whenever a pair of exploratory actions were chosen.
        self.current_exploratory_actions = None
        self.all_training_rewards = []
        self.chosen_start_end_nodes = {}

    def should_validate(self, max_training_steps):
        return self.current_training_step % self.validation_interval == 0 or self.current_training_step == max_training_steps

    def validate(self, validation_data, batch_sampler):
        """
        Validate agent performance on a set of graphs.

        :param validation_data:
        :param batch_sampler:
        :return:
        """
        # Generate a new batch of graphs from the validation dataset
        data_batch = self.sample_batch(validation_data, batch_sampler)

        # 1. Measuring performance of each graph at the initial step
        # performances_before = [self.environment.calculate_reward(graph) for graph in data_batch]
        # 2. Run a simulation and getting performances
        performances_after = self.simulate_for_validation(data_batch)

        # Calculate mean improvement improvement
        # performance = np.mean(performances_after - performances_before)

        # neptune_logging.log_batch_validation_result(performance)

        # return np.mean(performances_after - performances_before)

    def train(self, train_data, validation_data, max_steps):
        """
        Train the agent's policy.

        FIXME: Think in a way to accept a DataLoader instead of a list of graph states. This is not trivial due to
               dynamic sampling. In fact, we need a batch sampler that changes over the batch generation process.

        :param validation_data:
        :param train_data:
        :param max_steps: Maximum number of training steps
        :return:
        """

        # Initializer a new auxiliary sampler of graphs. It allows to get a random batch of graph IDs from the training data
        train_data_batch_sampler = BatchSampler(train_data, batch_size=self.batch_size)
        validation_data_batch_sampler = BatchSampler(validation_data, batch_size=self.batch_size)

        self.eps_step = max_steps / self.eps_step_denominator

        for i in range(1):
            self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                                          * (self.eps_step - max(0., self.current_training_step)) / self.eps_step)
            self.simulate_for_training(graphs=deepcopy(train_data))

        # Initialize Adam optimizer
        optimizer = optim.Adam(self.q_networks.parameters(), lr=self.learning_rate)
        for self.current_training_step in range(max_steps):
            # It's time to select a start node.
            self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                                          * (self.eps_step - max(0., self.current_training_step)) / self.eps_step)

            log.info(f'Training Step: {self.current_training_step} | Epsilon: {self.eps}')
            with torch.no_grad():
                # Sample a batch of data
                data_batch = self.sample_batch(train_data, train_data_batch_sampler)
                # Run a simulation. Oue agent is getting experience. No policy update at this stage.
                self.simulate_for_training(data_batch, perform_logging=True)

            # Check if it's time to update the target network
            if self.current_training_step % self.target_network_copy_interval == 0:
                self.update_target_networks()

            if self.should_validate(max_training_steps=max_steps):
                self.validate(validation_data, validation_data_batch_sampler)

            # Sample a batch of replays
            action_mode, states, actions, rewards, finished, next_states = self.experience_buffers.sample(
                self.batch_size)

            # Cast all rewards to tensor
            rewards_tensor = torch.Tensor(rewards).view(-1, 1)

            if USE_CUDA == 1:
                # Move rewards tensor to GPU whenever available
                rewards_tensor = rewards_tensor.cuda()

            # Keeps track on the next states that did not reach the end
            not_finished_next_states = []
            # Keeps track on the next states that did not reach the end
            not_finished_indexes = []
            for i in range(len(states)):
                # Iterate over all graphs and keep track on those not finished yet
                if not finished[i]:
                    # Store graph state
                    not_finished_next_states += [next_states[i]]
                    # Store the corresponding index
                    not_finished_indexes += [i]

            # print(len(not_finished_next_states))

            # Check whether at least one graph did not reach a final state
            if len(not_finished_next_states) > 0:
                # Get forbidden actions for each graph
                _, _, forbidden_actions = zip(*not_finished_next_states)
                # We are processing the next state, and then we need to know to which action it stands for
                next_action_mode = self.action_modes[(action_mode + 1) % len(self.action_modes)]
                # Get the q-value for the next state
                with torch.no_grad():
                    _, q_t_next, prefix_sum_next = self.target_q_networks(next_action_mode, not_finished_next_states,
                                                                          None)
                    # The previous network is returning the q-value for all existing actions. Now we need to filter out every
                    # forbidden action and choose the action with the highest q-value
                    _, q_rhs = self.target_q_networks.select_action_from_q_values(next_action_mode, q_t_next,
                                                                                  prefix_sum_next, forbidden_actions)

                # Save reward
                rewards_tensor[not_finished_indexes] = q_rhs * 0.99 + rewards_tensor[not_finished_indexes]

            # Convert reward to (len(graph_list), 1) shape
            rewards_tensor = Variable(rewards_tensor.view(-1, 1))

            # print("Next: ", rewards_tensor)

            # Get q-value for the current state
            _, q_s_all, _ = self.q_networks(action_mode, states, actions)

            actions_tensor = torch.tensor(actions).unsqueeze(-1)
            if USE_CUDA == 1:
                actions_tensor = actions_tensor.cuda()

            q_sa = q_s_all.gather(1, actions_tensor)

            if USE_CUDA == 1:
                rewards_tensor.cuda()

            # print("Current Q-Value: ", q_sa)

            # Calculate loss and gradients. Back-Propagate gradients
            loss = nn.MSELoss()(q_sa, rewards_tensor)

            print("Loss: ", loss)

            neptune_logging.log_batch_training_result(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def update_target_networks(self):
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())

    def pick_random_action(self, graph: GraphState):
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

    def choose_exploratory_actions(self):
        """
        Choose a random action. Note that,
        :return:
        """
        selected_start_nodes = []
        selected_end_nodes = []

        for graph in self.environment.graphs_list:
            start_node, end_node = self.pick_random_action(graph)

            selected_start_nodes += [start_node]
            selected_end_nodes += [end_node]

        return selected_start_nodes, selected_end_nodes

    def choose_greedy_actions(self):
        """
        Choose an action from the DQN model
        :return:
        """

        # Get the environments state of each graphs
        cur_env_states = self.environment.current_state
        # Get action that maximizes Q-value (for each graph)
        actions, q_values, _ = self.q_networks(action_mode=self.current_action_mode, states=cur_env_states,
                                               actions=None,
                                               greedy_acts=True)
        actions = list(actions.view(-1).cpu().numpy())

        return actions

    def choose_actions(self, execute_exploratory_actions=False):
        """
        Choose the next action given the current state of graph env. Note that we are working with batches. We call
        the Q-Network using a batch of graphs whenever we want to act greedily.

        Apart from greedy actions, random actions (that promote exploration) are taken with a certain probability.

        This method is the same across distinct action modes. Whenever ypu call it, the action mode defined in current_action_mode
        property determines the Q-Network to be used.
        :return: Returns a 1-D array with the best action for each graph. If GraphEnv contains a batch of 10 graphs, this method
        return a (10, 1) array
        """

        if not execute_exploratory_actions:
            return self.choose_greedy_actions()

        if self.current_action_mode == ACTION_MODE_SELECTING_START_NODE:
            # print("Selecting start node")

            if np.random.random() < self.eps:
                # Select an exploratory action
                selected_start_nodes, selected_end_nodes = self.choose_exploratory_actions()
                self.current_exploratory_actions = (selected_start_nodes, selected_end_nodes)

                # print("Choosing random actions: ", self.current_exploratory_actions)

                return selected_start_nodes
            else:
                # Choose a greedy action using the DQN
                self.current_exploratory_actions = None

                greedy_actions = self.choose_greedy_actions()
                # print("Choosing greedy action: ", greedy_actions)

                return greedy_actions
        else:
            # print("Selecting end node")
            # A start node is already selected. It's now time to select the end node
            if self.current_exploratory_actions is not None:
                # print("A previous end node exists: ", self.current_exploratory_actions[1])
                # If an exploratory action was chosen in the previous step, we already know the next action. It is stored
                # in the current_exploratory_actions
                return self.current_exploratory_actions[1]
            else:
                # Choose an end node from the DQN
                greedy_actions = self.choose_greedy_actions()
                # print("Choosing greedy action: ", greedy_actions)

                return greedy_actions

    def simulate_for_training(self, graphs, perform_logging=False):
        """
        Start a fresh simulation using the train_graphs. The agent plays N environments simultaneously,
        storing the result of each step in an experience buffer.

        :param perform_logging:
        :param graphs:
        :return:
        """
        # Init simulation environments with the current train_graphs
        self.environment.init(graphs)

        # This selector allows switching between actions modes by using next() (like a Python iterator)
        action_mode_selector = itertools.cycle(self.action_modes)

        # Inti current time step to 0
        time_step = 0
        while True:
            # Set the current action mode
            self.current_action_mode = next(action_mode_selector)
            # Decide the next action. Both greedy and exploratory actions are considered.
            actions = self.choose_actions(execute_exploratory_actions=True)

            graphs_before = [(graph, graph.selected_start_node, graph.forbidden_actions) for graph in
                             deepcopy(self.environment.graphs_list)]

            # Execute actions and step forward
            self.environment.step(actions)

            graphs_states = [(graph, graph.selected_start_node, graph.forbidden_actions) for graph in
                             self.environment.graphs_list]

            # print(f"Training Step: {self.current_training_step} | Simulation Step: {time_step} | Action Mode: {self.current_action_mode} | Action: {actions} | Reward: {np.mean(self.environment.rewards)}")

            if self.current_action_mode == ACTION_MODE_SELECTING_END_NODE:
                self.all_training_rewards += [np.mean(self.environment.rewards)]
                if perform_logging:
                    neptune_logging.log_training_instant_mean_reward(self.all_training_rewards[-1])
                    neptune_logging.log_training_mean_reward(np.mean(self.all_training_rewards[-100:]))

            if self.environment.is_terminal():
                dummy_final_next_states = [(None, None, None) for _ in range(len(graphs))]
                experience_buffer = self.experience_buffers.get_experience_buffer(self.current_action_mode)
                experience_buffer.append_many(graphs_states, actions, self.environment.rewards, [True] * len(actions),
                                              dummy_final_next_states)
                break
            else:
                experience_buffer = self.experience_buffers.get_experience_buffer(self.current_action_mode)
                experience_buffer.append_many(graphs_before, actions, self.environment.rewards,
                                              [False] * len(graphs_states),
                                              graphs_states)

            # Increment time step
            time_step += 1

        # Simulation is over! Now we need to store the final states in the experience buffer.
        rewards = self.environment.rewards

        return rewards

    def simulate_for_validation(self, graphs):
        print("Validating")

        # Init simulation environments with the current train_graphs
        self.environment.init(graphs)

        # This selector allows switching between actions modes by using next() (like a Python iterator)
        action_mode_selector = itertools.cycle(self.action_modes)

        rewards = np.zeros(len(graphs))

        # Inti current time step to 0
        time_step = 0
        while not self.environment.is_terminal():
            # Set the current action mode
            self.current_action_mode = next(action_mode_selector)

            # Decide the next action. Both greedy and exploratory actions are considered.
            actions = self.choose_actions(execute_exploratory_actions=False)

            # Execute actions and step forward
            self.environment.step(actions)

            rewards = self.environment.rewards

            # Log rewards
            if self.current_action_mode == ACTION_MODE_SELECTING_END_NODE:
                for graph_idx in range(len(self.environment.graphs_list)):
                    NEPTUNE_INSTANCE[f'validation/simulation/{self.current_training_step}/{graph_idx}/reward'].log(
                        rewards[graph_idx])
                NEPTUNE_INSTANCE[f'validation/simulation/{self.current_training_step}/reward_average'].log(
                    np.mean(rewards))
                NEPTUNE_INSTANCE[f'validation/simulation/{self.current_training_step}/reward_std'].log(np.std(rewards))

            # Increment time step
            time_step += 1

        NEPTUNE_INSTANCE[f'validation/simulation/{self.current_training_step}/final_reward_mean'].log(np.mean(rewards))
        NEPTUNE_INSTANCE[f'validation/simulation/{self.current_training_step}/final_reward_std'].log(np.std(rewards))

        # Save image of the first graph
        fig, ax = plt.subplots()
        draw_nx_graph_with_coordinates(self.environment.graphs_list[0].nx_graph, ax)
        neptune_logging.upload_graph_plot(fig, self.current_training_step)

        # Save irrigation and sources for the first graphs
        if self.environment.last_irrigation_map is not None and self.environment.last_sources is not None:
            fig_irrigation, ax_irrigation = plt.subplots()
            ax_irrigation.imshow(np.flip(self.environment.last_irrigation_map), cmap='hot', interpolation='nearest')
            fig_sources, ax_sources = plt.subplots()
            ax_sources.imshow(np.flip(self.environment.last_sources), cmap='hot', interpolation='nearest')
            neptune_logging.upload_irrigation_heatmaps(fig_sources, fig_irrigation, self.current_training_step,
                                                       'validation')

        # Compute statistics (insertion and removal frequency)
        actions_stats = self.environment.action_type_statistics
        for graph_idx, stat in enumerate(actions_stats):
            fig, ax = plt.subplots()
            ax.hist(stat, bins=[0, 0.8, 1, 1.8])
            neptune_logging.upload_action_frequency(fig, self.current_training_step, graph_idx)

        fig, axs = plt.subplots(2)
        axs[0].bar(self.environment.start_node_selection_statistics.keys(),
                   self.environment.start_node_selection_statistics.values(), 2, color='g')
        axs[1].bar(self.environment.end_node_selection_statistics.keys(),
                   self.environment.end_node_selection_statistics.values(), 2, color='g')
        neptune_logging.upload_action_selection(fig, self.current_training_step)

        return rewards

    @staticmethod
    def sample_batch(data, data_sampler):
        # Generate a batch of graph indexes for training data purposes
        batch_indices = data_sampler.sample()
        # Get training data from previously generated indexes
        data_batch = [deepcopy(data[idx]) for idx in batch_indices]

        return data_batch
