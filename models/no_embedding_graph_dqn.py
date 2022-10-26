import numpy as np
import torch
from torch import nn
from torch.nn import Linear

from settings import USE_CUDA


class NoEmbeddingGraphDQN(nn.Module):

    def __init__(self, unique_id: int, input_dim: int, hidden_output_dim: int, actions_output_dim: int) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            Linear(input_dim, hidden_output_dim),
            nn.ReLU(),
            Linear(hidden_output_dim, actions_output_dim)
        )

        if USE_CUDA == 1:
            self.fc = self.fc.cuda()

        self.unique_id = unique_id

    @staticmethod
    def select_action_from_q_values(q_values, forbidden_actions):
        q_values = q_values.data.clone()

        min_tensor = torch.tensor(float(np.finfo(np.float32).min)).type_as(q_values)
        forbidden_actions_bool = forbidden_actions.bool()

        q_values[forbidden_actions_bool] = min_tensor

        values, indices = torch.topk(q_values, 1, dim=1)

        return indices, values

    @staticmethod
    def strip_forbidden_actions(states):
        num_nodes = int(states[0][0].item())
        forbidden_actions = states[:, 1: num_nodes + 1]
        new_states = states[:, num_nodes + 2:]

        return new_states, forbidden_actions

    def prepare_data(self, states):
        return self.strip_forbidden_actions(states)

    def forward(self, states):
        graph_representation, forbidden_actions = self.prepare_data(states)

        print(f"Unique ID: {self.unique_id} | Graph Representation Size: {graph_representation.shape}")

        q_values = self.fc(graph_representation)

        return q_values, forbidden_actions
