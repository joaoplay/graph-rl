from abc import abstractmethod, ABC

import numpy as np


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


class IrrigationThresholdAchieved(StopCondition):
    
    def is_satisfied(self, environment):
        irrigation_map = environment.last_irrigation_map

        if irrigation_map is None:
            return False

        sections_x = np.array_split(irrigation_map, 20, axis=0)
        sections_y = np.array_split(irrigation_map, 20, axis=1)

        satisfied_x = all([np.all(section > 0.0) for section in sections_x])
        satisfied_y = all( [np.all(section > 0.0) for section in sections_y])

        return satisfied_x and satisfied_y
