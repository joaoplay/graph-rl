from abc import abstractmethod, ABC

import numpy as np
import torch


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

        sections_x = torch.tensor_split(irrigation_map, 20, dim=0)
        sections_y = torch.tensor_split(irrigation_map, 20, dim=1)

        satisfied_x = all([torch.all(section > 0.5) for section in sections_x])
        satisfied_y = all([torch.all(section > 0.5) for section in sections_y])

        # print(f"Sections X Satisfied: {satisfied_x} | Sections Y Satisfied: {satisfied_y}")

        return satisfied_x and satisfied_y
