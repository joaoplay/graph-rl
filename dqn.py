from argparse import Namespace
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from agents.graph_agent import GraphAgent
from agents.replay_memory.multi_action_replay_buffer import MultiActionReplayBuffer
from agents.rl_dataset import RLDataset
from environments.graph_env import GraphEnv, DEFAULT_ACTION_MODES, ACTION_MODE_SELECTING_START_NODE, \
    ACTION_MODE_SELECTING_END_NODE
from models.multi_action_mode_dqn import MultiActionModeDQN
from settings import NEPTUNE_INSTANCE
from util import draw_nx_irrigation_network
from neptune.new.types import File


class DQN:

    def __init__(self, env: GraphEnv = None, graphs=None, batch_size: int = 32, lr: float = 0.00025,
                 gamma: float = 0.99, sync_rate: int = 10000, replay_size: int = 10 ** 6,
                 eps_last_frame: int = 5 * 10 ** 5, eps_start: float = 1.0, eps_end: float = 0.2,
                 warm_start_steps: int = 50000, action_modes: tuple[int] = DEFAULT_ACTION_MODES,
                 multi_action_q_network: dict = None, num_dataloader_workers: int = 1, device='cpu') -> None:
        super().__init__()

        self.hparams = Namespace(
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            sync_rate=sync_rate,
            replay_size=replay_size,
            eps_last_frame=eps_last_frame,
            eps_start=eps_start,
            eps_end=eps_end,
            warm_start_steps=warm_start_steps,
            action_modes=action_modes,
            multi_action_q_network=multi_action_q_network,
            num_dataloader_workers=num_dataloader_workers,
            device=device
        )

        number_of_nodes = graphs[0].num_nodes

        # FIXME: This is hardcoded for now. Should be changed to a more general solution.
        start_representation_dim = graphs[0].start_node_selection_representation_dim + (
            env.compressed_irrigation_matrix_size if env.inject_irrigation else 0) + 1  # Inject goal

        self.q_networks = MultiActionModeDQN(action_modes=action_modes,
                                             input_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: start_representation_dim,
                                                 ACTION_MODE_SELECTING_END_NODE: start_representation_dim,
                                             },
                                             action_output_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: number_of_nodes,
                                                 ACTION_MODE_SELECTING_END_NODE: 8,
                                             }, **multi_action_q_network).to(device)
        self.target_q_networks = MultiActionModeDQN(action_modes=action_modes,
                                                    input_dim={
                                                        ACTION_MODE_SELECTING_START_NODE: start_representation_dim,
                                                        ACTION_MODE_SELECTING_END_NODE: start_representation_dim,
                                                    },
                                                    action_output_dim={
                                                        ACTION_MODE_SELECTING_START_NODE: number_of_nodes,
                                                        ACTION_MODE_SELECTING_END_NODE: 8,
                                                    }, **multi_action_q_network).to(device)

        self.env = env
        self.graphs = graphs
        self.buffer = MultiActionReplayBuffer(action_modes)
        self.agent = GraphAgent(self.env, self.graphs, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.global_step = 0
        self.device = device
        self.optimizer = optim.Adam(self.q_networks.parameters(), lr=self.hparams.lr)
        self.total_validation_wins = 0
        self.total_validation_losses = 0
        self.total_training_wins = 0
        self.total_training_losses = 0

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            reward, done, _ = self.agent.play_step(self.q_networks, epsilon=1.0)
            NEPTUNE_INSTANCE['training/instant_reward'].log(reward)

            self.episode_reward += reward
            if done:
                NEPTUNE_INSTANCE['training/cum_reward'].log(self.episode_reward)
                self.episode_reward = 0

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        action_modes, states, actions, rewards, solved, next_states, goals = batch

        # NEPTUNE_INSTANCE['training/batch_mean_reward'].log(torch.mean(rewards))
        # NEPTUNE_INSTANCE['training/batch_std_reward'].log(torch.std(rewards))
        # NEPTUNE_INSTANCE['training/batch_min_reward'].log(torch.min(rewards))
        # NEPTUNE_INSTANCE['training/batch_max_reward'].log(torch.max(rewards))

        action_mode = int(action_modes[1].item())
        actions_tensor = actions.unsqueeze(-1)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions_tensor = actions_tensor.to(self.device)
        rewards = rewards.to(self.device)
        solved = solved.to(self.device)

        state_action_values, _ = self.q_networks(action_mode, states)

        q_sa = state_action_values.gather(1, actions_tensor)

        rewards = rewards.unsqueeze(-1)

        not_solved = ~solved
        if torch.any(not_solved):
            with torch.no_grad():
                not_done_next_states = next_states[not_solved]
                next_action_mode = (action_mode + 1) % len(self.hparams.action_modes)

                # Get the q-value for the next state
                next_state_values, forbidden_actions = self.target_q_networks(next_action_mode, not_done_next_states)
                _, not_done_next_station_action_values = self.target_q_networks.select_action_from_q_values(
                    next_action_mode, next_state_values, forbidden_actions)
                expected_state_action_values = not_done_next_station_action_values * self.hparams.gamma + rewards[
                    not_solved]

                rewards[not_solved] = expected_state_action_values

        return action_mode, nn.MSELoss()(q_sa, rewards)

    def play_step(self):
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step / self.hparams.eps_last_frame,
        )

        # Step through environment with agent
        reward, done, solved = self.agent.play_step(self.q_networks, epsilon, self.device)
        self.episode_reward += reward

        # Log step results
        NEPTUNE_INSTANCE['training/instant_reward'].log(reward)
        NEPTUNE_INSTANCE['training/epsilon'].log(epsilon)

        return reward, done, solved

    def train_batch(self, batch):
        action_mode, loss = self.dqn_mse_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if action_mode == ACTION_MODE_SELECTING_START_NODE:
            NEPTUNE_INSTANCE['training/start-node-selection-loss'].log(loss)
        else:
            NEPTUNE_INSTANCE['training/end-node-selection-loss'].log(loss)

        NEPTUNE_INSTANCE['training/goal'].log(torch.mean(batch[6]))

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            #print(f"Syncing target networks at step {self.global_step}")
            self.target_q_networks.load_state_dict(self.q_networks.state_dict())

        return action_mode, loss

    def train(self, steps: int, validation_interval: int) -> None:
        dataset = RLDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)

        self.global_step = 0

        # Start training loop
        for step in range(steps):
            if step % validation_interval == 0:
                self.validate()

            # Play a step
            reward, done, solved = self.play_step()

            # Get a single batch of data from the replay buffer
            batch = next(iter(dataloader))
            # Train the network on the batch
            self.train_batch(batch)

            if done:
                # The episode has ended. Log the episode reward and reset it
                NEPTUNE_INSTANCE['training/cum_reward'].log(self.episode_reward)
                self.total_reward = self.episode_reward
                self.episode_reward = 0

                if not solved:
                    # The previous episode did not solve the environment.
                    for t in range(self.env.max_steps):
                        batch = next(iter(dataloader))
                        self.train_batch(batch)

            self.global_step += 1

    def validate(self):
        """Tests the agent in the environment.

        """
        #print("Validating...")

        validation_env = deepcopy(self.env)
        validation_agent = GraphAgent(validation_env, self.graphs, self.buffer)
        validation_agent.reset()

        done = False
        cum_reward = 0
        while not done:
            reward, done, solved = validation_agent.play_validation_step(self.q_networks, self.device)

            cum_reward += reward
            NEPTUNE_INSTANCE[f'validation/{self.global_step}/instant-reward'].log(reward)

        NEPTUNE_INSTANCE[f'validation/episode-length'].log(validation_agent.env.steps_counter)

        fig, axs = plt.subplots(2)
        axs[0].bar(validation_agent.env.start_node_selection_statistics.keys(),
                   validation_agent.env.start_node_selection_statistics.values(), 2, color='g')
        axs[1].bar(validation_agent.env.end_node_selection_statistics.keys(),
                   validation_agent.env.end_node_selection_statistics.values(), 2, color='g')
        NEPTUNE_INSTANCE[f'validation/{self.global_step}/action-selection'].log(File.as_image(fig))

        if validation_agent.env.last_irrigation_map is not None:
            fig_irrigation, ax_irrigation = plt.subplots()
            ax_irrigation.title.set_text(f'Global Step: {validation_agent.total_steps}')
            ax_irrigation.imshow(np.flipud(validation_agent.env.last_irrigation_map), cmap='hot', vmin=0,
                                 interpolation='nearest')

            NEPTUNE_INSTANCE[f'validation/{self.global_step}/irrigation'].log(File.as_image(fig_irrigation))

        if validation_agent.env.last_irrigation_graph is not None and validation_agent.env.last_pressures is not None \
                and validation_agent.env.last_edge_sources is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.title.set_text(f'Global Step: {validation_agent.total_steps}')
            draw_nx_irrigation_network(validation_agent.env.last_irrigation_graph, validation_agent.env.last_pressures,
                                       validation_agent.env.last_edge_sources, validation_agent.env.last_edges_list, ax)
            NEPTUNE_INSTANCE[f'validation/{self.global_step}/network-debug'].log(File.as_image(fig))

        plt.close('all')

        return {'episode-length': validation_agent.env.steps_counter}












