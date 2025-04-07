from ptseries.models import PTLayer
from ptseries.models.observables import Observable
import torch.nn as nn
import torch
import numpy as np


class ORCALayer(nn.Module):
    """ ORCA Layer class """

    def __init__(self,
                 in_features: int,
                 observable: str | Observable = "avg-photons",
                 gradient_mode: str = "parameter-shift",
                 gradient_delta: float = np.pi / 10,
                 n_samples: int = 100,
                 n_tiling: int = 1,
                 tbi_type: str | None = None,
                 n_loops: int | None = None,
                 url: str | None = None,
                 **tbi_params) -> None:
        super().__init__()
        input_state = [1]*in_features
        self.pt_layer = PTLayer(input_state=input_state, in_features=in_features, observable=observable, gradient_mode=gradient_mode,
                                gradient_delta=gradient_delta, n_samples=n_samples, tbi_params=tbi_params | {"tbi_type": tbi_type, "n_loops": n_loops, "url": url}, n_tiling=n_tiling)

    def forward(self, x: torch.Tensor | None = None, n_samples: int | None = None):
        return self.pt_layer.forward(x, n_samples)

    def set_thetas(self, theta_values: torch.Tensor):
        self.pt_layer.set_thetas(theta_values)
