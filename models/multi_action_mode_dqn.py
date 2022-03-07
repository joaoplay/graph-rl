from typing import Optional

from torch import nn

from models.graph_dqn import GraphDQN
from models.noisy_graph_dqn import NoisyGraphDQN


class MultiActionModeDQN(nn.Module):
    """
    Graph construction is a  two-step process: 1) the start node of a new edge is selected; 2) the end node of that edge
    is selected.

    The MultiActionModeDQN allows defining multiple DQN, one for each type of action. In the context of graph construction,
    a DQN is responsible for the start node selection policy whereas the second one selects the end node.
    """

    def __init__(self, action_modes: tuple[int], embedding_dim: Optional[dict], hidden_output_dim: Optional[dict],
                 num_node_features: Optional[dict], action_output_dim: Optional[dict]):
        super().__init__()

        self._dqn_by_action_mode = nn.ModuleDict(
            {str(action_mode): NoisyGraphDQN(unique_id=action_mode,
                                             embedding_dim=embedding_dim[action_mode],
                                             hidden_output_dim=hidden_output_dim[action_mode],
                                             num_node_features=num_node_features[action_mode],
                                             actions_output_dim=action_output_dim[action_mode])
             for action_mode in action_modes})

    def select_action_from_q_values(self, action_mode, q_t_next, prefix_sum_next, forbidden_actions):
        """
        Select an action from a set of Q-Values. Actions are indexed by its position in the Q-Values list. For instance,
        the index 0 in the Q-values list matches the selection the node with ID 0.
        :param action_mode:
        :param q_t_next:
        :param prefix_sum_next:
        :param forbidden_actions:
        :return:
        """
        return self._dqn_by_action_mode[str(action_mode)].select_action_from_q_values(q_t_next, prefix_sum_next,
                                                                                      forbidden_actions)

    def forward(self, action_mode, states, actions, greedy_acts=False):
        action_mode = str(action_mode)
        return self._dqn_by_action_mode[action_mode](states, actions, greedy_acts)
