import numpy as np

from graphs.edge_budget.edge_budget import BaseEdgeBudget
from graphs.graph_state import GraphState


class FixedEdgePercentageBudget(BaseEdgeBudget):

    def __init__(self, graphs: list[GraphState], max_node_degree: int = 8, fixed_edge_percentage: float = 0.4) -> None:
        self.fixed_edge_percentage = fixed_edge_percentage
        self.max_node_degree = max_node_degree
        super().__init__(graphs)

    def init_budget(self, graphs: list[GraphState]):
        budget_list = [int(graph.num_nodes) for graph in graphs]
        self.budget = np.asarray(budget_list)
