import numpy as np

from graphs.edge_budget.base_edge_budget import BaseEdgeBudget
from graphs.graph_state import GraphState


class FixedEdgePercentageBudget(BaseEdgeBudget):

    def __init__(self, graphs: list[GraphState], fixed_edge_percentage: float = 0.4) -> None:
        self.fixed_edge_percentage = fixed_edge_percentage
        super().__init__(graphs)

    def init_budget(self, graphs: list[GraphState]):
        budget_list = [(int(graph.num_nodes) * 8 * self.fixed_edge_percentage) for graph in graphs]
        self.budget = np.asarray(budget_list)
