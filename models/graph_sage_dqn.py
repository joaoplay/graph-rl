import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from settings import USE_CUDA


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

        if USE_CUDA == 1:
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv1.cuda()
            self.fc = self.fc.cuda()

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
        graphs, selected_nodes, forbidden_actions = states

        pygeom_data = []
        for graph in graphs:
            pygeom_data += [graph.to_pygeom_representation()]

        data_loader = DataLoader(pygeom_data, batch_size=len(graphs))
        data = next(iter(data_loader))

        if USE_CUDA == 1:
            data.x = data.x.type(torch.FloatTensor).cuda()
            data.edge_index = data.edge_index.cuda()

        print(data.x.is_cuda)
        print(data.edge_index.is_cuda)

        conv1_res = self.conv1(data.x, data.edge_index)

        if USE_CUDA == 1:
            conv1_res = conv1_res.cuda()

        conv2_res = self.conv2(conv1_res, data.edge_index)

        grouped_conv2_res = torch.reshape(conv2_res, (len(graphs), graphs[0].num_nodes, self.embedding_dim))

        mean_embeddings = torch.mean(grouped_conv2_res, dim=1)

        q_values = self.fc(mean_embeddings)

        return q_values, forbidden_actions


