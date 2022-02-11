from abc import ABC, abstractmethod


class BaseGraphGenerator(ABC):

    def generate(self):
        graph = self.generate_graph()
        graph_state = self.from_graph_to_state(graph)
        return graph_state

    @abstractmethod
    def generate_graph(self):
        pass

    @abstractmethod
    def from_graph_to_state(self, gra):
        pass
