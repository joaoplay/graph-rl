from torch import nn


class GraphA2C(nn.Module):

    def __init__(self, input_shape, n_actions) -> None:
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_shape, ),
            nn.ReLU()
        )