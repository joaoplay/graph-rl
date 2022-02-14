from abc import abstractmethod, ABC


class StopCondition(ABC):

    @abstractmethod
    def is_satisfied(self, environment):
        pass


class MaxStepsExceeded(StopCondition):

    def __init__(self, max_steps) -> None:
        super().__init__()
        self.max_steps = max_steps

    def is_satisfied(self, environment):
        return environment.steps_counter > self.max_steps


class AllGraphsExhausted(StopCondition):

    def is_satisfied(self, environment):
        return environment.edges_budget.all_graphs_are_exhausted
