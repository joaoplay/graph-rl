from torch import nn


class GraphA2C(nn.Module):

    def __init__(self, graph_input_size, n_actions) -> None:
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(graph_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(graph_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.policy(x), self.value(x)
