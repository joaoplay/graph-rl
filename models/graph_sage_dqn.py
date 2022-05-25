import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv


class GraphSageDQN(nn.Module):

    def __init__(self, unique_id: int, embedding_dim: int, hidden_output_dim: int, num_node_features: int, actions_output_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_node_features = num_node_features

        self.conv1 = SAGEConv(in_channels=-1, out_channels=embedding_dim)
        self.conv2 = SAGEConv(in_channels=embedding_dim, out_channels=embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_output_dim),
            nn.ReLU(),
            nn.Linear(hidden_output_dim, actions_output_dim)
        )

    @staticmethod
    def select_action_from_q_values(q_values, forbidden_actions):
        q_values = q_values.data.clone()

        min_tensor = torch.tensor(float(np.finfo(np.float32).min)).type_as(q_values)

        forbidden_actions_bool = []
        for i in range(100):
            if i in forbidden_actions:
                forbidden_actions_bool.append(True)
            else:
                forbidden_actions_bool.append(False)

        q_values[forbidden_actions_bool] = min_tensor

        values, indices = torch.topk(q_values, 1, dim=1)

        return indices, values

    def forward(self, states):
        graphs, selected_nodes, forbidden_actions = states

        pygeom_data = []
        for graph in graphs:
            pygeom_data += [graph.to_pygeom_representation()]

        data_loader = DataLoader(pygeom_data)
        data = next(iter(data_loader))

        conv1_res = self.conv1(data.x.type(torch.FloatTensor), data.edge_index)
        conv2_res = self.conv2(conv1_res, data.edge_index)

        mean_embeddings = torch.mean(conv2_res, dim=0)
        q_values = self.fc(mean_embeddings)

        return q_values, forbidden_actions


