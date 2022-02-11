from abc import abstractmethod, ABC


class StopCondition(ABC):

    @abstractmethod
    def is_satisfied(self, environment):
        pass

