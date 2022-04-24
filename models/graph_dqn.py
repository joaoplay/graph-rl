import os
import sys

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from graph_embedding.graph_embedding import EmbedMeanField
from settings import USE_CUDA

if os.getenv('DEV_MODE_ENABLE', None):
    sys.path.append('graph_embedding/pytorch_structure2vec/s2v_lib')
else:
    sys.path.append('/usr/lib/pytorch_structure2vec/s2v_lib')

from pytorch_util import weights_init

# FIXME: This DQN definition was adpted from.... Add copyright

class GraphDQN(nn.Module):
    """
    Deep Q-Network that operates on graphs, adding edges between nodes in the graph. This module was built to support
    batch training, operating on a batch of graphs and corresponding state conditions.

    Note that this network is used for start node and end node selection.

    Architecture:
        1 - Pass Graph States (and selected nodes, when selection the end node) through a structure2vec module. It outputs
            an embedding representation of the whole graph and for each node.
        2 - For each node embedding, concatenate the embedding representation of the whole graph. Each node consists of
            embedding_size

    """

    def __init__(self, unique_id: int, embedding_dim: int, hidden_output_dim: int, num_node_features: int, actions_output_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_node_features = num_node_features

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_output_dim),
            nn.ReLU(),
            nn.Linear(hidden_output_dim, actions_output_dim)
        )

        self.unique_id = unique_id

        weights_init(self)

        # FIXME: The number of layers is hardcoded
        # self.s2v_model = SAGEConv(in_channels=1, out_channels=embedding_dim)
        self.s2v_model = EmbedMeanField(latent_dim=embedding_dim,
                                        output_dim=0,
                                        num_node_feats=self.num_node_features,
                                        num_edge_feats=0,
                                        max_lv=3)

    def prepare_node_features(self, graphs, selected_nodes):
        n_nodes = 0
        prefix_sum = []
        picked_ones = []
        coordinates = []
        for i in range(len(graphs)):
            coordinates += graphs[i].get_node_features()

            if selected_nodes is not None and selected_nodes[i] is not None:
                picked_ones.append(n_nodes + selected_nodes[i])
            n_nodes += graphs[i].num_nodes
            prefix_sum.append(n_nodes)

        node_features = torch.FloatTensor(coordinates)

        selected_node_encoding = torch.zeros(n_nodes, 2)
        selected_node_encoding[:, 0] = 1.0

        if len(picked_ones):
            selected_node_encoding.numpy()[picked_ones, 1] = 1.0
            selected_node_encoding.numpy()[picked_ones, 0] = 0.0

        node_features = torch.cat((node_features, selected_node_encoding), dim=1)

        return node_features, torch.LongTensor(prefix_sum)

    @staticmethod
    def select_action_from_q_values(q_values, prefix_sum_tensor, forbidden_actions):
        offset = 0
        banned_acts = []
        prefix_sum = prefix_sum_tensor.data.cpu().numpy()
        #no_action_available = torch.zeros(len(prefix_sum), dtype=torch.bool)
        for i in range(len(prefix_sum)):
            # Iterate all examples
            if forbidden_actions is not None and forbidden_actions[i] is not None:
                # Check for forbidden actions
                # if len(forbidden_actions[i]) == prefix_sum[0]:
                    # There are no available actions. This is a "no action" action
                #    no_action_available[i] = True

                # Store all forbidden actions. A very low q value will be assigned to those action in order to avoid
                # them to be selected.
                for j in forbidden_actions[i]:
                    banned_acts.append(offset + j)

            offset = prefix_sum[i]

        q_values = q_values.data.clone()
        q_values.resize_(len(q_values))

        banned = torch.LongTensor(banned_acts)
        if USE_CUDA == 1:
            banned = banned.cuda()

        if len(banned_acts):
            # Apply a very low q value to forbidden actions
            min_tensor = torch.tensor(float(np.finfo(np.float32).min))
            if USE_CUDA == 1:
                min_tensor = min_tensor.cuda()
            q_values.index_fill_(0, banned, min_tensor)

        jagged = q_values.reshape(len(prefix_sum), prefix_sum[0])

        #print(jagged)

        values, indices = torch.topk(jagged, 1, dim=1)

        #print(indices, values)

        # Assign a "no action" (-1) to every example with no available actions
        # indices[no_action_available] = -1

        return indices, values

    @staticmethod
    def rep_global_embed(graph_embed, sum_prefixes):
        prefix_sum = sum_prefixes.data.cpu().numpy()

        rep_idx = []
        for i in range(len(prefix_sum)):
            if i == 0:
                n_nodes = prefix_sum[i]
            else:
                n_nodes = prefix_sum[i] - prefix_sum[i - 1]
            rep_idx += [i] * n_nodes

        rep_idx = Variable(torch.LongTensor(rep_idx))
        if USE_CUDA == 1:
            rep_idx = rep_idx.cuda()
        graph_embed = torch.index_select(graph_embed, 0, rep_idx)
        return graph_embed

    @staticmethod
    def add_offset(actions, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        shifted = []
        for i in range(len(prefix_sum)):
            if i > 0:
                offset = prefix_sum[i - 1]
            else:
                offset = 0
            shifted.append(actions[i] + offset)

        return shifted

    def forward(self, states, actions, greedy_acts=False):
        graphs, selected_nodes, forbidden_actions = states

        node_features, prefix_sum = self.prepare_node_features(graphs, selected_nodes)

        if USE_CUDA == 1:
            node_features = node_features.cuda()
            prefix_sum = prefix_sum.cuda()

        embed, graph_embed = self.s2v_model(graphs, node_features, None)

        if actions is None:
            graph_embed = self.rep_global_embed(graph_embed, prefix_sum)
        else:
            shifted = self.add_offset(actions, prefix_sum)
            embed = embed[shifted, :]

        embed_s_a = torch.cat((embed, graph_embed), dim=1)
        raw_pred = self.fc(embed_s_a)
        if greedy_acts:
            actions, _ = self.select_action_from_q_values(raw_pred, prefix_sum, forbidden_actions)

        return actions, raw_pred, prefix_sum
