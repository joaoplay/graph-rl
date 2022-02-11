import numpy as np

from graphs.edge_budget.edge_budget import BaseEdgeBudget
from graphs.graph_state import GraphState


class InfiniteEdgeBudget(BaseEdgeBudget):
    """
    Unbounded edge budget
    """

    def init_budget(self, graphs: list[GraphState]):
        budget_list = [int(np.iinfo(np.int32).max) for graph in graphs]
        self.budget = np.asarray(budget_list)