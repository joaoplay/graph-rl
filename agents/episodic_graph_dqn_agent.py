import itertools
import logging
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable

import neptune_logging
from agents.graph_dqn_agent import GraphDQNAgent
from agents.util.sample_tracker import BatchSampler
from environments.graph_env import GraphEnv, DEFAULT_ACTION_MODES, ACTION_MODE_SELECTING_END_NODE
from settings import USE_CUDA

log = logging.getLogger(__name__)


class EpisodicGraphDQNAgent(GraphDQNAgent):

    def __init__(self, environment: GraphEnv, start_node_selection_dqn_params: Dict,
                 end_node_selection_dqn_params: Dict, batch_size: int = 50, warm_up: int = 4,
                 learning_rate: float = 0.0001, eps_start: float = 1, eps_step_denominator: float = 2,
                 eps_end: float = 0.1, validation_interval: int = 1000, target_network_copy_interval: int = 50,
                 action_modes: tuple[int] = DEFAULT_ACTION_MODES) -> None:
        super().__init__(environment, start_node_selection_dqn_params, end_node_selection_dqn_params, batch_size,
                         warm_up, learning_rate, eps_start, eps_step_denominator, eps_end, validation_interval,
                         target_network_copy_interval, action_modes)

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

            # Check whether at least one graph did not reach a final state
            if len(not_finished_next_states) > 0:
                # Get forbidden actions for each graph
                _, _, forbidden_actions = zip(*not_finished_next_states)
                # We are processing the next state, and then we need to know to which action it stands for
                next_action_mode = self.action_modes[(action_mode + 1) % len(self.action_modes)]

                with torch.no_grad():
                    # Get the q-value for the next state
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

            # Get q-value for the current state
            _, q_s_all, _ = self.q_networks(action_mode, states, actions)

            actions_tensor = torch.tensor(actions).unsqueeze(-1)
            if USE_CUDA == 1:
                actions_tensor = actions_tensor.cuda()

            q_sa = q_s_all.gather(1, actions_tensor)

            if USE_CUDA == 1:
                rewards_tensor.cuda()

            # Calculate loss and gradients. Back-Propagate gradients
            loss = nn.MSELoss()(q_sa, rewards_tensor)

            print("Loss: ", loss)

            neptune_logging.log_batch_training_result(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

            dummy_final_next_states = [(None, None, None) for _ in range(len(graphs))]
            experience_buffer = self.experience_buffers.get_experience_buffer(self.current_action_mode)
            if self.environment.all_graphs_exhausted():  # Irrigation problem was solved
                rewards = [1]
                experience_buffer.append_many(graphs_states, actions, rewards, [True] * len(actions),
                                              dummy_final_next_states)

                self.all_training_rewards += rewards

                if perform_logging:
                    print("Reward: ", rewards)
                    self.log_reward()

                return  # Episode has finished
            elif self.environment.is_terminal():  # Irrigation problem was not solved
                rewards = [-1]
                experience_buffer.append_many(graphs_states, actions, rewards, [True] * len(actions),
                                              dummy_final_next_states)

                self.all_training_rewards += rewards

                if perform_logging:
                    print("Reward: ", rewards)
                    self.log_reward()

                return  # Episode has finished
            else:
                rewards = [0]
                experience_buffer.append_many(graphs_before, actions, rewards,
                                              [False] * len(graphs_states),
                                              graphs_states)

                self.all_training_rewards += rewards

                if perform_logging:
                    self.log_reward()

            # Increment time step
            time_step += 1

    def log_reward(self, ):
        neptune_logging.log_training_instant_mean_reward(self.all_training_rewards[-1])
        neptune_logging.log_training_mean_reward(np.mean(self.all_training_rewards[-100:]))