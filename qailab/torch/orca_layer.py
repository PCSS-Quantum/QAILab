""" ORCA layer module """
from typing import Literal
from ptseries.models import PTLayer
from torch import nn
import torch
import numpy as np

from qailab.orca_api.pt_adapter import PTAdapter


class ORCALayer(nn.Module):
    """Neural Network layer using ORCA quantum computers, a PTLayer wrapper

    Parameters
    ----------
    in_features: int,
        number fo layer input features
    observable: Literal['avg-photons', 'covariances', 'correlations'], default = 'avg-photons'
        method of conversion from measurements to a tensor. Default is "avg-photons".
    gradient_mode: Literal['parameter-shift', 'finite-difference', 'spsa'], default = 'parameter-shift',
        method to compute the gradient. Default is "parameter-shift".
    gradient_delta: float, default = np.pi / 10
        Delta to use with the parameter shift rule or for the finite difference. Default is np.pi / 10.
    n_samples: int, default = 100
        Number of samples to draw. Default is 100.
    n_tiling: int, default = 1
        Uses n_tiling instances of PT Series and concatenates the results.
        Input features are distributed between tiles, which each have different trainable params. Default is 1.
    tbi_type: Literal['multi-loop', 'single-loop', 'fixed-random-unitary', 'PT'], default = 'single-loop'
        Type of TBI to return. Can be 'multi-loop', 'single-loop', 'fixed-random-unitary' or 'PT.
        Choose PT to run on real PT device. Default is 'single-loop'.
    n_loops: int, default = 1
        Number of loops in the TBI. Default it 1.
    url: str, default = None
        The URL of the PT device, for example "http://<orca_api_address>".
    tbi_params: dict, default = None
        Dictionary of optional parameters to instantiate the TBI. Default is None.
    """

    def __init__(self,
                 in_features: int,
                 observable: Literal['avg-photons', 'covariances', 'correlations'] = "avg-photons",
                 gradient_mode: Literal['parameter-shift', 'finite-difference', 'spsa'] = 'parameter-shift',
                 gradient_delta: float = np.pi / 10,
                 n_samples: int = 100,
                 n_tiling: int = 1,
                 tbi_type: str | None = None,
                 n_loops: int | None = None,
                 url: str | None = None,
                 machine: str | None = None,
                 secret_key: str | None = None,
                 **tbi_params) -> None:
        super().__init__()
        input_state = list(map(lambda x: x % 2, range(in_features + 1)))
        self.pt_layer = PTLayer(input_state=input_state, in_features=in_features, observable=observable, gradient_mode=gradient_mode,
                                gradient_delta=gradient_delta, n_samples=n_samples,
                                tbi_params=tbi_params | {"tbi_type": tbi_type, "n_loops": n_loops, "url": url}, n_tiling=n_tiling)
        if tbi_type == "PT":
            if url is None or machine is None or secret_key is None:
                raise ValueError('Parameters url, machine and secret key need to be provided to use real PT device')
            if n_loops is None:
                n_loops = 1
            self.pt_layer.tbi = PTAdapter(n_loops=n_loops, url=url, machine=machine, secret_key=secret_key, **tbi_params)

    def forward(self, x: torch.Tensor | None = None, n_samples: int | None = None) -> torch.Tensor:
        """ORCA layer pytorch Module forward method

        Parameters
        ----------
        x: torch.Tensor | None, default = None.
            layer input tensor
        n_samples: int | None, default = None.
            number of samples from quantum computer

        Returns
        -------
        output: torch.Tensor
            layer output tensor
        """
        return self.pt_layer.forward(x, n_samples)  # type: ignore

    def set_thetas(self, theta_values: torch.Tensor):
        """set theta parameters of layer's beam splitter gates

        Parameters
        ----------
        theta_values: torch.Tensor
            values of theta parameter of layer's beam splitter gates
        """
        self.pt_layer.set_thetas(theta_values)
