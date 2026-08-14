import warnings

warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

import torch
import torch.nn as nn

from flappy_env import ACTION_SIZE, STATE_SIZE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE),
        ).to(DEVICE)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)
