from copy import deepcopy, copy
from itertools import compress
from typing import Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from neptune.new.types import File
from torch import nn

from agents.replay_memory.multi_action_replay_buffer import MultiActionReplayBuffer
from agents.util.sample_tracker import BatchSampler
from environments.graph_env import GraphEnv, ACTION_MODE_SELECTING_START_NODE, ACTION_MODE_SELECTING_END_NODE
from graphs.graph_state import GraphState
from models.multi_action_mode_dqn import MultiActionModeDQN
from settings import NEPTUNE_INSTANCE, USE_CUDA
from util import draw_nx_irrigation_network


class GraphAgent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(self, env: GraphEnv, graph_list, replay_buffer: MultiActionReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.total_steps = 0
        self.replay_buffer = replay_buffer
        self.graph_list = graph_list
        self.batch_sampler = BatchSampler(self.graph_list, batch_size=1)
        self.exploratory_actions_cache = None
        self.state = None
        self.wins = 0
        self.looses = 0
        self.selected_start_nodes_stats = {}
        self.selected_end_nodes_stats = {}
        self.episode_reward = 0
        self.q_values_history = []
        self.current_episode_transitions = None
        self.reset()

    def reset(self):
        data_batch = self.sample_batch(self.graph_list, self.batch_sampler)
        self.env.init(data_batch)
        self.state = self.env.current_graph_representation_copy
        self.selected_start_nodes_stats = {}
        self.selected_end_nodes_stats = {}
        self.episode_reward = 0
        self.q_values_history = []
        self.current_episode_transitions = {action_mode: [] for action_mode in self.replay_buffer.action_modes}

    def choose_greedy_actions(self, action_mode, q_network):
        """
        Choose an action from the DQN model
        :return:
        """

        # Get the environments state of each graphs
        state = torch.tensor(self.state)

        # FIXME: Pass it as a parameter
        goal = np.ones((1, 1))
        concat_state_goal = np.concatenate([state, goal], axis=1)
        state_plus_goal = torch.tensor(concat_state_goal, dtype=torch.float).to(state.device)

        if USE_CUDA == 1:
            state_plus_goal = state_plus_goal.cuda()

        # Get action that maximizes Q-value (for each graph)
        q_values, forbidden_actions = q_network(action_mode=action_mode, states=state_plus_goal)
        actions, _ = q_network.select_action_from_q_values(action_mode=action_mode, q_values=q_values,
                                                           forbidden_actions=forbidden_actions)
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

                return greedy_actions
        else:
            # A start node is already selected. It's now time to select the end node
            if self.exploratory_actions_cache is not None:
                # If an exploratory action was chosen in the previous step, we already know the next action. It is stored
                # in the current_exploratory_actions
                return self.exploratory_actions_cache[1]
            else:
                # Choose an end node from the DQN
                greedy_actions = self.choose_greedy_actions(action_mode=action_mode, q_network=q_network)

                return greedy_actions

    @torch.no_grad()
    def play_validation_step(self, q_networks: MultiActionModeDQN, device: str = "cpu"):
        actions = self.get_action(self.env.current_action_mode, q_networks, 0.0, device)
        # Do step in the environment
        new_state, reward, done, solved = self.env.step(actions)

        self.state = new_state

        return reward[0], done[0], solved[0]

    @torch.no_grad()
    def play_step(self, q_networks: MultiActionModeDQN, epsilon: float = 0.0, device: str = "cpu"):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, solved
        """

        actions = self.get_action(self.env.current_action_mode, q_networks, epsilon, device)

        previous_action_mode = self.env.current_action_mode
        previous_solved = deepcopy(self.env.solved)

        # Do step in the environment
        new_state, reward, done, solved = self.env.step(actions)

        self.episode_reward += reward[0]

        now_solved = [True for idx, was_solved in enumerate(previous_solved) if not was_solved and solved[idx]]
        not_solved = [True for idx, was_solved in enumerate(previous_solved) if not was_solved and not solved[idx]]

        prev_states = []
        next_states = []
        rewards = []
        all_solved = []

        if any(now_solved):
            prev_states += list(compress(self.state, now_solved))
            next_states += list(compress(new_state, now_solved))
            now_solved_rewards = list(compress(reward, now_solved))
            rewards += now_solved_rewards
            all_solved += [True] * len(rewards)
            self.wins += sum([1 for _ in now_solved_rewards if self.env.steps_counter < self.env.max_steps])
            self.looses += sum([1 for _ in now_solved_rewards if self.env.steps_counter >= self.env.max_steps])

        if any(not_solved):
            prev_states += list(compress(self.state, not_solved))
            next_states += list(compress(new_state, not_solved))
            rewards += list(compress(reward, not_solved))
            all_solved += [False] * len(rewards)

        if len(prev_states) > 0:
            experiences = self.replay_buffer.append_many(action_mode=previous_action_mode, states=prev_states,
                                                         actions=actions, rewards=rewards, terminals=all_solved,
                                                         next_states=next_states, goals=[1 for _ in range(len(prev_states))])
            self.current_episode_transitions[previous_action_mode].extend(experiences)

        # Log statistics
        if previous_action_mode == ACTION_MODE_SELECTING_START_NODE:
            if not actions[0] in self.selected_start_nodes_stats:
                self.selected_start_nodes_stats[actions[0]] = 0

            self.selected_start_nodes_stats[actions[0]] += 1
        else:
            if not actions[0] in self.selected_end_nodes_stats:
                self.selected_end_nodes_stats[actions[0]] = 0

            self.selected_end_nodes_stats[actions[0]] += 1

        self.state = new_state
        if all(done):
            print(
                f"Current Simulation Step: {self.env.steps_counter} | Win: {self.wins} | Looses: {self.looses} | Episode Reward: {self.episode_reward}")

            if not self.env.irrigation_goal_achieved():
                irrigation_score = self.env.previous_irrigation_score
                for act_mode, action_mode_experiences in self.current_episode_transitions.items():
                    self.replay_buffer.append_many(action_mode=act_mode,
                                                   states=[e.state[:-1] for e in action_mode_experiences],
                                                   actions=[e.action for e in action_mode_experiences],
                                                   rewards=[e.reward for e in action_mode_experiences],
                                                   terminals=[e.solved for e in action_mode_experiences],
                                                   next_states=[e.new_state[:-1] for e in action_mode_experiences],
                                                   goals=[irrigation_score[0] for _ in range(len(action_mode_experiences))])
            # Reset environment
            self.reset()

        self.total_steps += 1

        return reward[0], done[0], solved[0]

    @staticmethod
    def sample_batch(data, data_sampler):
        # Generate a batch of graph indexes for training data purposes
        batch_indices = data_sampler.sample()
        # Get training data from previously generated indexes
        data_batch = [deepcopy(data[idx]) for idx in batch_indices]

        return data_batch
