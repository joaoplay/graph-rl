from torch import nn

from models.graph_dqn import GraphDQN
from models.noisy_linear import NoisyLinear
from settings import USE_CUDA


class NoisyGraphDQN(GraphDQN):

    def __init__(self, unique_id: int, embedding_dim: int, hidden_output_dim: int, num_node_features: int,
                 actions_output_dim: int) -> None:
        super().__init__(unique_id, embedding_dim, hidden_output_dim, num_node_features, actions_output_dim)

        self.noisy_layers = [
            NoisyLinear(embedding_dim * 2, hidden_output_dim),
            NoisyLinear(hidden_output_dim, actions_output_dim)
        ]

        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

        if USE_CUDA == 1:
            self.fc.cuda()

