import math

import torch
from torch import nn
from torch.nn.functional import linear

from settings import USE_CUDA


class NoisyLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, sigma_init=0.017, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)

        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
        self.sigma_bias = nn.Parameter(w)
        z = torch.zeros(out_features)

        self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

        if USE_CUDA == 1:
            self.sigma_bias.cuda()
            self.sigma_weight.cuda()
            self.epsilon_weight.cuda()
            self.bias.cuda()
            self.weight.cuda()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight

        return linear(x, v, bias)
