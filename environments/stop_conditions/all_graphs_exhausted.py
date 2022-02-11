from environments.stop_conditions.stop_condition import StopCondition


class AllGraphsExhausted(StopCondition):

    def is_satisfied(self, environment):
        return environment.edges_budget.all_graphs_are_exhausted
