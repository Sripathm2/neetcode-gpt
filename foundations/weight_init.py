import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        w = torch.empty(fan_out, fan_in)
        nn.init.xavier_normal_(w)
        return w.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        w = torch.empty(fan_out, fan_in)
        nn.init.kaiming_normal_(w)
        return w.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.

        torch.manual_seed(0)

        # Step 1: build all weight matrices first
        weights = []
        fan_in = input_dim
        for i in range(num_layers):
            fan_out = hidden_dim
            w = torch.empty(fan_out, fan_in)
            if init_type == 'xavier':
                nn.init.xavier_normal_(w)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(w)
            elif init_type == 'random':
                nn.init.normal_(w)  # plain N(0,1), not kaiming
            weights.append(w)
            fan_in = fan_out

        # Step 2: draw the input after all weights are created
        x = torch.randn(input_dim)

        # Step 3: forward pass using the pre-built weights
        return_std = []
        for w in weights:
            x = w @ x
            x = torch.relu(x)
            return_std.append(round(x.std().item(), 2))

        return return_std
