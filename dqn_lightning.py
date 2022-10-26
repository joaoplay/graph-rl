from collections import OrderedDict
from copy import deepcopy
from typing import Any, Tuple, List

import numpy as np
import torch
#import wandb
from matplotlib import pyplot as plt
from neptune.new.types import File
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch import nn

from agents.graph_agent import GraphAgent
from agents.replay_memory.multi_action_replay_buffer import MultiActionReplayBuffer
from agents.rl_dataset import RLDataset
from environments.graph_env import DEFAULT_ACTION_MODES, ACTION_MODE_SELECTING_START_NODE, \
    ACTION_MODE_SELECTING_END_NODE, GraphEnv
from models.multi_action_mode_dqn import MultiActionModeDQN
from settings import NEPTUNE_INSTANCE
from util import draw_nx_irrigation_network


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(self, env: GraphEnv = None, graphs=None, batch_size: int = 32, lr: float = 0.00025,
                 gamma: float = 0.99, sync_rate: int = 10000, replay_size: int = 10 ** 6,
                 eps_last_frame: int = 5 * 10 ** 5, eps_start: float = 1.0, eps_end: float = 0.2,
                 warm_start_steps: int = 50000, action_modes: tuple[int] = DEFAULT_ACTION_MODES,
                 multi_action_q_network: dict = None, num_dataloader_workers: int = 1) -> None:
        super().__init__()

        self.save_hyperparameters()

        number_of_nodes = graphs[0].num_nodes
        start_representation_dim = graphs[0].start_node_selection_representation_dim
        end_representation_dim = graphs[0].end_node_selection_representation_dim


        self.q_networks = MultiActionModeDQN(action_modes=self.hparams.action_modes,
                                             input_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: start_representation_dim,
                                                 ACTION_MODE_SELECTING_END_NODE: end_representation_dim,
                                             },
                                             action_output_dim={
                                                 ACTION_MODE_SELECTING_START_NODE: number_of_nodes,
                                                 ACTION_MODE_SELECTING_END_NODE: 8,
                                             }, **self.hparams.multi_action_q_network)
        self.target_q_networks = MultiActionModeDQN(action_modes=self.hparams.action_modes,
                                                    input_dim={
                                                        ACTION_MODE_SELECTING_START_NODE: start_representation_dim,
                                                        ACTION_MODE_SELECTING_END_NODE: end_representation_dim,
                                                    },
                                                    action_output_dim={
                                                        ACTION_MODE_SELECTING_START_NODE: number_of_nodes,
                                                        ACTION_MODE_SELECTING_END_NODE: 8,
                                                    }, **self.hparams.multi_action_q_network)

        #wandb.watch(self.q_networks, log="all")
        #wandb.watch(self.target_q_networks, log="all")

        self.env = env
        self.graphs = graphs
        self.buffer = MultiActionReplayBuffer(self.hparams.action_modes)
        self.agent = GraphAgent(self.env, self.graphs, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            reward, done = self.agent.play_step(self.q_networks, epsilon=1.0)
            NEPTUNE_INSTANCE['training/instant_reward'].log(reward)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        # FIXME: This is not used for training. Broken now!
        output = self.q_networks(x)
        return output

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        action_modes, states, actions, rewards, dones, next_states = batch

        NEPTUNE_INSTANCE['training/batch_mean_reward'].log(torch.mean(rewards))
        NEPTUNE_INSTANCE['training/batch_std_reward'].log(torch.std(rewards))
        NEPTUNE_INSTANCE['training/batch_min_reward'].log(torch.min(rewards))
        NEPTUNE_INSTANCE['training/batch_max_reward'].log(torch.max(rewards))

        NEPTUNE_INSTANCE['training/batch_total_dones'].log(torch.sum(dones))

        action_mode = int(action_modes[1].item())
        actions_tensor = actions.unsqueeze(-1)

        state_action_values, _ = self.q_networks(action_mode, states)

        q_sa = state_action_values.gather(1, actions_tensor)

        rewards = rewards.unsqueeze(-1)

        not_dones = ~dones
        if torch.any(not_dones):
            with torch.no_grad():
                not_done_next_states = next_states[not_dones]
                next_action_mode = (action_mode + 1) % len(self.hparams.action_modes)

                # Get the q-value for the next state
                next_state_values, forbidden_actions = self.target_q_networks(next_action_mode, not_done_next_states)
                _, not_done_next_station_action_values = self.target_q_networks.select_action_from_q_values(
                    next_action_mode, next_state_values, forbidden_actions)
                expected_state_action_values = not_done_next_station_action_values * self.hparams.gamma + rewards[
                    not_dones]

                rewards[not_dones] = expected_state_action_values

        return action_mode, nn.MSELoss()(q_sa, rewards)

    def training_step(self, batch, nb_batch):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        # FIXME: Confirm it!
        self.q_networks.to(device)
        self.target_q_networks.to(device)

        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step / self.hparams.eps_last_frame,
        )

        # Step through environment with agent
        reward, done = self.agent.play_step(self.q_networks, epsilon, device)
        self.episode_reward += reward

        NEPTUNE_INSTANCE['training/instant_reward'].log(reward)

        #wandb.log({'epsilon': epsilon})

        NEPTUNE_INSTANCE['training/epsilon'].log(epsilon)

        # NEPTUNE_INSTANCE['training/total_wins'].log(self.agent.wins)
        # NEPTUNE_INSTANCE['training/total_looses'].log(self.agent.looses)

        # Calculates training loss
        action_mode, loss = self.dqn_mse_loss(batch)

        if action_mode == ACTION_MODE_SELECTING_START_NODE:
            NEPTUNE_INSTANCE['training/start-node-selection-loss'].log(loss)
        else:
            NEPTUNE_INSTANCE['training/end-node-selection-loss'].log(loss)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            print(f"Syncing target networks at step {self.global_step}")
            self.target_q_networks.load_state_dict(self.q_networks.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        return {"loss": loss, "log": log, "progress_bar": status}

    def validation_step(self, batch, ncb_batch):
        """Tests the agent in the environment.

        """
        validation_env = deepcopy(self.env)
        validation_agent = GraphAgent(validation_env, self.graphs, self.buffer)
        validation_agent.reset()

        done = False
        while not done:
            reward, done = validation_agent.play_validation_step(self.q_networks, self.get_device(batch))

            NEPTUNE_INSTANCE[f'validation/{self.current_epoch}/instant-reward'].log(reward)

        NEPTUNE_INSTANCE[f'validation/episode-length'].log(validation_agent.env.steps_counter)

        fig, axs = plt.subplots(2)
        axs[0].bar(validation_agent.env.start_node_selection_statistics.keys(),
                   validation_agent.env.start_node_selection_statistics.values(), 2, color='g')
        axs[1].bar(validation_agent.env.end_node_selection_statistics.keys(),
                   validation_agent.env.end_node_selection_statistics.values(), 2, color='g')
        NEPTUNE_INSTANCE[f'validation/{self.current_epoch}/action-selection'].log(File.as_image(fig))

        if validation_agent.env.last_irrigation_map is not None:
            fig_irrigation, ax_irrigation = plt.subplots()
            ax_irrigation.title.set_text(f'Global Step: {validation_agent.total_steps}')
            ax_irrigation.imshow(np.flipud(validation_agent.env.last_irrigation_map), cmap='hot', vmin=0,
                                 interpolation='nearest')

            NEPTUNE_INSTANCE[f'validation/{self.current_epoch}/irrigation'].log(File.as_image(fig_irrigation))

        if validation_agent.env.last_irrigation_graph is not None and validation_agent.env.last_pressures is not None \
                and validation_agent.env.last_edge_sources is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.title.set_text(f'Global Step: {validation_agent.total_steps}')
            draw_nx_irrigation_network(validation_agent.env.last_irrigation_graph, validation_agent.env.last_pressures,
                                       validation_agent.env.last_edge_sources, validation_agent.env.last_edges_list, ax)
            NEPTUNE_INSTANCE[f'validation/{self.current_epoch}/network-debug'].log(File.as_image(fig))

        plt.close('all')

    def print_network_params(self):
        print("Q-Networks")
        for name, param in self.q_networks.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        print("Target Q-Networks")
        for name, param in self.target_q_networks.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.q_networks.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


def graph_collate_fn(batch):
    return batch
