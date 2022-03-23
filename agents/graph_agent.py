from copy import deepcopy
from itertools import compress
from typing import Tuple, List

import numpy as np
import torch
from torch import nn

from agents.replay_memory.multi_action_replay_buffer import MultiActionReplayBuffer
from agents.util.sample_tracker import BatchSampler
from environments.graph_env import GraphEnv, ACTION_MODE_SELECTING_START_NODE
from models.multi_action_mode_dqn import MultiActionModeDQN


class GraphAgent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(self, env: GraphEnv, graph_list, replay_buffer: MultiActionReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.graph_list = graph_list
        self.batch_sampler = BatchSampler(self.graph_list, batch_size=1)
        self.exploratory_actions_cache = None
        self.state = None
        self.reset()

    def reset(self):
        data_batch = self.sample_batch(self.graph_list, self.batch_sampler)
        self.env.init(data_batch)
        self.state = self.env.current_state_copy

    def choose_greedy_actions(self, action_mode, q_network):
        """
        Choose an action from the DQN model
        :return:
        """

        # Get the environments state of each graphs
        state = self.env.current_state
        # Get action that maximizes Q-value (for each graph)
        actions, q_values, _ = q_network(action_mode=action_mode, states=state, actions=None, greedy_acts=True)
        actions = list(actions.view(-1).cpu().numpy())

        return actions

    def get_action(self, action_mode: int, q_network: nn.Module, epsilon: float, device: str) -> List[int]:
        """

        :param action_mode:
        :param q_network:
        :param epsilon:
        :param device:
        :return:
        """

        if action_mode == ACTION_MODE_SELECTING_START_NODE:
            if np.random.random() < epsilon:
                # Select an exploratory action
                selected_start_nodes, selected_end_nodes = self.env.get_random_actions()
                self.exploratory_actions_cache = (selected_start_nodes, selected_end_nodes)

                return selected_start_nodes
            else:
                # Choose a greedy action using the DQN
                self.exploratory_actions_cache = None

                greedy_actions = self.choose_greedy_actions(action_mode=action_mode, q_network=q_network)
                # print("Choosing greedy action: ", greedy_actions)

                return greedy_actions
        else:
            # print("Selecting end node")
            # A start node is already selected. It's now time to select the end node
            if self.exploratory_actions_cache is not None:
                # print("A previous end node exists: ", self.current_exploratory_actions[1])
                # If an exploratory action was chosen in the previous step, we already know the next action. It is stored
                # in the current_exploratory_actions
                return self.exploratory_actions_cache[1]
            else:
                # Choose an end node from the DQN
                greedy_actions = self.choose_greedy_actions(action_mode=action_mode, q_network=q_network)
                # print("Choosing greedy action: ", greedy_actions)

                return greedy_actions

    @torch.no_grad()
    def play_step(self, q_networks: MultiActionModeDQN, epsilon: float = 0.0, device: str = "cpu"):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        actions = self.get_action(self.env.current_action_mode, q_networks, epsilon, device)

        previous_action_mode = self.env.current_action_mode
        previous_done = deepcopy(self.env.done)

        # Do step in the environment
        new_state, reward, done = self.env.step(actions)

        now_done = [True for idx, was_done in enumerate(previous_done) if not was_done and done[idx]]
        not_done = [True for idx, was_done in enumerate(previous_done) if not was_done and not done[idx]]

        prev_states = []
        next_states = []
        rewards = []
        all_done = []
        if any(now_done):
            prev_states += compress(self.state, now_done)
            next_states += compress(new_state, now_done)
            rewards += compress(reward, now_done)
            all_done += [True] * len(rewards)

        if any(not_done):
            prev_states += compress(self.state, not_done)
            next_states += compress(new_state, not_done)
            rewards += compress(reward, not_done)
            all_done += [False] * len(rewards)

        if len(prev_states) > 0:
            self.replay_buffer.append_many(action_mode=previous_action_mode, states=prev_states,
                                           actions=actions, rewards=rewards, terminals=all_done,
                                           next_states=next_states)

        self.state = new_state
        if all(done):
            self.reset()

        return reward[0], done[0]

    @staticmethod
    def sample_batch(data, data_sampler):
        # Generate a batch of graph indexes for training data purposes
        batch_indices = data_sampler.sample()
        # Get training data from previously generated indexes
        data_batch = [deepcopy(data[idx]) for idx in batch_indices]

        return data_batch