import torch

from models.no_embedding_graph_dqn import NoEmbeddingGraphDQN


class EndNodeSelectionDQN(NoEmbeddingGraphDQN):

    def __init__(self, unique_id: int, embedding_dim: int, hidden_output_dim: int, num_node_features: int,
                 actions_output_dim: int) -> None:
        super().__init__(unique_id, embedding_dim, hidden_output_dim, num_node_features, actions_output_dim)

    @staticmethod
    def select_action_from_q_values(q_values, prefix_sum_tensor, forbidden_actions):
        q_values = q_values.data.clone()
        q_values = torch.flatten(q_values)

        prefix_sum = prefix_sum_tensor.data.cpu().numpy()
        jagged = q_values.reshape(len(prefix_sum), prefix_sum[0])

        values, indices = torch.topk(jagged, 1, dim=1)

        return indices, values

    def forward(self, states, actions, greedy_acts=False):
        actions, raw_pred, prefix_sum = super().forward(states, actions, greedy_acts)

        return actions


