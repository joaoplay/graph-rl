from abc import ABC, abstractmethod
import numpy as np

from graphs.graph_state import GraphState


class BaseEdgeBudget(ABC):
    """
    Base class for an edge budget. A maximum budget is set at the simulation kickoff and updated as new edges are created

    The init_budget() method determines how the budget of edges for each graph.
    """

    def __init__(self, graphs: list[GraphState]) -> None:
        super().__init__()
        self.budget = np.zeros(len(graphs), dtype=np.float)
        self.used = np.zeros(len(graphs), dtype=np.float)
        self.forced_exhausting = [False] * len(graphs)
        self.init_budget(graphs)

    @abstractmethod
    def init_budget(self, graphs: list[GraphState]):
        pass

    def is_exhausted(self, graph_idx):
        """
        Checks whether a given graph is already exhausted.
        :param graph_idx: The graph index must match the order specified in the graphs list.
        :return:
        """
        return (self.budget[graph_idx] - self.used[graph_idx]) == 0 or self.forced_exhausting[graph_idx]

    @property
    def all_non_exhausted(self):
        """
        Return all non-exhausted graphs.
        :return: List of graphs indexes
        """
        return [idx for idx in range(len(self.budget)) if not self.is_exhausted(idx)]

    @property
    def all_exhausted(self):
        """
        Return all exhausted graphs.
        :return: List of graphs indexes
        """
        return [idx for idx in range(len(self.budget)) if self.is_exhausted(idx)]

    @property
    def all_graphs_are_exhausted(self):
        """
        Returns True when all graphs are exhausted.
        :return:
        """
        return all([self.is_exhausted(idx) for idx in range(len(self.budget))])

    def get_remaining_budget(self, graph_idx):
        return self.budget[graph_idx] - self.used[graph_idx]

    def increment_used_budget(self, graph_idx, value):
        self.used[graph_idx] += value

    def force_exhausting(self, graph_idx):
        self.forced_exhausting[graph_idx] = True
