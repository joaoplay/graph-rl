import numpy as np
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv

from settings import USE_CUDA


class GraphSageDQN(nn.Module):

    def __init__(self, unique_id: int, embedding_dim: int, hidden_output_dim: int, num_node_features: int, actions_output_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_node_features = num_node_features

        self.conv1 = GCNConv(in_channels=3, out_channels=embedding_dim)
        self.conv2 = GCNConv(in_channels=embedding_dim, out_channels=embedding_dim)
        self.fc = Linear(in_features=embedding_dim, out_features=actions_output_dim)

        """self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_output_dim),
            nn.ReLU(),
            nn.Linear(hidden_output_dim, actions_output_dim)
        )"""

        if USE_CUDA == 1:
            self.conv1 = self.conv1.cuda(device=1)
            self.conv2 = self.conv2.cuda(device=1)
            self.fc = self.fc.cuda(device=1)

    @staticmethod
    def select_action_from_q_values(q_values, forbidden_actions):
        q_values = q_values.data.clone()

        min_tensor = torch.tensor(float(np.finfo(np.float32).min)).type_as(q_values)

        forbidden_actions_mask = torch.zeros(q_values.shape, dtype=torch.bool)
        for g_idx, g_forbidden_actions in enumerate(forbidden_actions):
            forbidden_actions_mask[g_idx, list(g_forbidden_actions)] = True

        q_values[forbidden_actions_mask] = min_tensor

        values, indices = torch.topk(q_values, 1, dim=1)

        return indices, values

    def forward(self, states):
        graphs, selected_nodes, forbidden_actions, graphs_pyg = states

        #pygeom_data = []
        #for graph in graphs:
        #    pygeom_data += [graph.to_pygeom_representation]

        data_loader = DataLoader(graphs_pyg, batch_size=len(graphs))
        data = next(iter(data_loader))

        if USE_CUDA == 1:
            data.x = data.x.type(torch.FloatTensor).cuda(device=1)
            data.edge_index = data.edge_index.cuda(device=1)
            data.batch = data.batch.cuda(device=1)
        else:
            data.x = data.x.type(torch.FloatTensor)

        conv1_res = self.conv1(data.x, data.edge_index)
        conv1_res = nn.ReLU()(conv1_res)
        conv2_res = self.conv2(conv1_res, data.edge_index)

        graph_embed = global_mean_pool(conv2_res, data.batch)

        q_values = self.fc(graph_embed)

        return q_values, forbidden_actions


