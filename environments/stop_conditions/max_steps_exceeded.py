from environments.stop_conditions.stop_condition import StopCondition


class MaxStepsExceeded(StopCondition):

    def __init__(self, max_steps) -> None:
        super().__init__()
        self.max_steps = max_steps

    def is_satisfied(self, environment):
        return environment.steps_counter > self.max_steps
