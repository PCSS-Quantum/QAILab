""" Module with QModel """
from collections.abc import Callable
from typing import Literal
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import mlflow
from .qmodel import QModel
try:
    from ptseries.optimizers import HybridOptimizer
except ImportError:
    class HybridOptimizer():
        """Dummy HO"""
        # pylint: disable=too-few-public-methods

        def __init__(
            self,
            model,
            lr_classical=0.01,
            lr_quantum=0.01,
            optimizer_quantum='SGD',
            optimizer_classical='Adam',
            betas=(0.9, 0.999),
            spsa_resamplings=1,
            spsa_gamma_decay=0.101,
            spsa_alpha_decay=0.602
        ):
            pass


class MLFlowQModel(QModel):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        module: nn.Module,
        loss: Callable,
        optimizer_type: type[Optimizer] | type[HybridOptimizer] | str = 'adamw',
        learning_rate: float | Literal['auto'] = 'auto',
        quantum_learning_rate: float | Literal['auto'] = 'auto',
        batch_size: int = 1,
        epochs: int = 1,
        validation_fraction: float = 0.2,
        shuffle: bool = True,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        metric: Literal["accuracy", "mse"] | None = None
    ):
        super().__init__(module, loss, optimizer_type, learning_rate, quantum_learning_rate,
                         batch_size, epochs, validation_fraction, shuffle, device, metric)
        self._epoch = 0

    def fit(self, x: Tensor | np.ndarray | pd.DataFrame, y: Tensor | np.ndarray | pd.DataFrame | pd.Series) -> "QModel":
        """ scikit-learn like fit method
        trains the neural network based on training set (x,y).

        Parameters
        ----------
        x: Tensor | np.ndarray | pd.DataFrame
            The training input samples of shape (n_samples, n_features).
        y: Tensor | np.array | pd.DataFrame | pd.Series
            The training target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self: QModel
            trained NN model
        """
        with mlflow.start_run():
            mlflow.log_params({k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, str, bool))})
            res = super().fit(x, y)
            mlflow.pytorch.log_model(self.module.cpu(), "model")
        return res

    def set_mlflow(self, experiment_name: str):
        """ set name and parameters """
        mlflow.set_experiment(experiment_name)

    def _train_one_epoch(self, train_loader: DataLoader) -> tuple[np.floating, np.floating]:
        loss, metrics = super()._train_one_epoch(train_loader)
        mlflow.log_metric("train_loss", float(loss), step=self._epoch)
        return loss, metrics

    def _validate_one_epoch(self, validation_loader: DataLoader) -> tuple[np.floating, np.floating]:
        loss, metrics = super()._validate_one_epoch(validation_loader)
        mlflow.log_metric("val_loss", float(loss), step=self._epoch)
        self._epoch += 1
        return loss, metrics
