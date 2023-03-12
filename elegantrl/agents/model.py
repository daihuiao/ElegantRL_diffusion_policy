# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from elegantrl.agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(t_dim+16, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.mid_layer1 = nn.Sequential(nn.Linear(action_dim, 256),
                                        nn.Mish(),
                                        nn.Linear(256, 256),
                                        nn.Mish(),
                                        nn.Linear(256, 16),
                                        nn.Mish())
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.mid_layer1(x)
        x = torch.cat([x, t], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)
