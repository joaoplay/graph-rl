import numpy as np
import torch

from graphs.graph_state import GraphState
from models.no_embedding_graph_dqn import NoEmbeddingGraphDQN


class NoEmbeddingGraphDQNEnd(NoEmbeddingGraphDQN):

    @staticmethod
    def select_action_from_q_values(q_values, forbidden_actions, graph_state: GraphState):
        q_values = q_values.data.clone()

        min_tensor = torch.tensor(float(np.finfo(np.float32).min)).type_as(q_values)

        neighbours = graph_state.nx_graph.neighbors(graph_state.selected_start_node)
        neighbours_size = len(neighbours)

        # Invalidate non-existing neighbours
        q_values[neighbours_size:] = min_tensor





