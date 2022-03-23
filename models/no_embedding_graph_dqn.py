import numpy as np
import torch
from torch import nn
from torch.nn import Linear

from graphs.graph_state import GraphState
from settings import USE_CUDA


class NoEmbeddingGraphDQN(nn.Module):

    def __init__(self, unique_id: int, embedding_dim: int, hidden_output_dim: int, num_node_features: int, actions_output_dim: int) -> None:
        super().__init__()

        self.num_node_features = num_node_features

        self.fc = nn.Sequential(
            #Linear(784, hidden_output_dim),
            Linear(169, hidden_output_dim),
            nn.ReLU(),
            Linear(hidden_output_dim, actions_output_dim)
        )

        self.unique_id = unique_id

    @staticmethod
    def select_action_from_q_values(q_values, prefix_sum_tensor, forbidden_actions):
        offset = 0
        banned_acts = []
        prefix_sum = prefix_sum_tensor.data.cpu().numpy()
        for i in range(len(prefix_sum)):
            # Iterate all examples
            if forbidden_actions is not None and forbidden_actions[i] is not None:

                # Store all forbidden actions. A very low q value will be assigned to those action in order to avoid
                # them to be selected.
                for j in forbidden_actions[i]:
                    banned_acts.append(offset + j)

            offset = prefix_sum[i]

        q_values = q_values.data.clone()
        q_values = torch.flatten(q_values)

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

        jagged_np = jagged.numpy()

        values, indices = torch.topk(jagged, 1, dim=1)

        return indices, values

    def convert_graph_to_one_hot_representation(self, graph: GraphState):
        nx_graph = graph.nx_graph.to_undirected()
        nx_neighbourhood_graph = graph.nx_neighbourhood_graph

        graph_one_hot = []
        for node in nx_graph.nodes:
            number_of_neighbours = nx_neighbourhood_graph.degree[node]
            all_neighbours = nx_neighbourhood_graph.neighbors(node)
            neighbours = nx_graph.neighbors(node)
            neighbours = sorted(neighbours)
            neighbours_one_hot = np.zeros(number_of_neighbours)
            existing_neighbours = [node in neighbours for node in all_neighbours]
            neighbours_one_hot[existing_neighbours] = 1
            graph_one_hot.append(neighbours_one_hot)

        flatten_graph_one_hot = np.concatenate(graph_one_hot).ravel()

        return flatten_graph_one_hot

    def prepare_data(self, graphs, selected_nodes, forbidden_actions):
        final_graphs = []
        prefix_sum = []
        previous_prefix_sum = 0
        for graph_idx, graph in enumerate(graphs):
            graph_one_hot = self.convert_graph_to_one_hot_representation(graph)
            selected_node_one_hot = np.zeros(graph.nx_graph.number_of_nodes())

            selected_node = selected_nodes[graph_idx]
            if selected_node is not None:
                selected_node_one_hot[selected_node] = 1

            graph_representation = np.concatenate((selected_node_one_hot, graph_one_hot))

            final_graphs += [torch.from_numpy(graph_representation)]
            prefix_sum += [previous_prefix_sum + graph.nx_graph.number_of_nodes()]
            previous_prefix_sum += graph.nx_graph.number_of_nodes()

        return torch.stack(final_graphs).float(), torch.tensor(prefix_sum)

    def forward(self, states, actions, greedy_acts=False):
        graphs, selected_nodes, forbidden_actions = zip(*states)

        encoded_graphs, prefix_sum = self.prepare_data(graphs, selected_nodes, forbidden_actions)

        if USE_CUDA == 1:
            encoded_graphs = encoded_graphs.cuda()

        raw_pred = self.fc(encoded_graphs)

        if greedy_acts:
            actions, _ = self.select_action_from_q_values(raw_pred, prefix_sum, forbidden_actions)

        return actions, raw_pred, prefix_sum
