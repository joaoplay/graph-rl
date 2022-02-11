from abc import ABC, abstractmethod

from environments.graph_env import GraphEnv


class BaseAgent(ABC):

    def __init__(self, environment: GraphEnv) -> None:
        super().__init__()
        self.environment = environment

    @abstractmethod
    def train(self, train_data, validation_data, max_steps):
        pass

    def validate(self, validation_data, batch_sampler):
        pass

    @abstractmethod
    def choose_actions(self):
        pass
