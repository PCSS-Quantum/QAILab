""" Module with QModel """
from collections.abc import Callable
from typing import Literal
from sklearn.base import BaseEstimator
import torch
from torch import Tensor, optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import pandas as pd

AVAILABLE_OPTIMIZERS: dict[str, type[Optimizer]] = {opt.__name__.lower(): opt for opt in [
    optim.Adam, optim.AdamW, optim.SGD, optim.Adadelta, optim.Adagrad, optim.Adamax, optim.RMSprop, optim.Rprop, optim.LBFGS]}


class QModel(BaseEstimator):
    """ Quantum model class

    Parameters
    ----------
    module: nn.Module
        pytorch Module representing the quantum or classical neural network.
    loss: Callable
        pytorch loss function to be used during training.
    optimizer_type: type[Optimizer] | str, default = "adamw"
        pytorch Optimizer class to be used during training.
    learning_rate: float | Literal['auto'], default = "auto"
        learning rate used by the optimizer, "auto" sets it to optimizer's default one.
    batch_size: int, default = 1
        number of training examples in batch.
    epochs: int, default = 1
        number of epochs to train the model.
    validation_fraction: float, default = 0.2
       share of the training dataset to be used for validation.
    shuffle: bool, default = True
        whether to shuffle data every epoch.
    device: {"cpu","cuda","mps"}, default="cpu"
        the device neural network will be trained on.

    Attributes
    ----------
    optimizer: Optimizer
        pytorch optimizer object used during training
    """
    module: nn.Module
    loss: Callable
    optimizer_type: type[Optimizer]
    optimizer: Optimizer
    learning_rate: float | Literal['auto']
    batch_size: int
    epochs: int
    validation_fraction: float
    shuffle: bool
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    def __init__(
        self,
        module: nn.Module,
        loss: Callable,
        optimizer_type: type[Optimizer] | str = 'adamw',
        learning_rate: float | Literal['auto'] = 'auto',
        batch_size: int = 1,
        epochs: int = 1,
        validation_fraction: float = 0.2,
        shuffle: bool = True,
        device: Literal["cpu", "cuda", "mps"] = "cpu"
    ):
        super().__init__()
        self.module = module
        self.loss = loss
        if isinstance(optimizer_type, str):
            if optimizer_type not in AVAILABLE_OPTIMIZERS:
                raise ValueError(
                    f"Unknown optimizer: {optimizer_type}. Available optimizers are: {list(AVAILABLE_OPTIMIZERS.keys())}")
            optimizer_type = AVAILABLE_OPTIMIZERS[optimizer_type]
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        if self.learning_rate == 'auto':
            self.optimizer = self.optimizer_type(self.module.parameters())  # type: ignore
        else:
            self.optimizer = self.optimizer_type(self.module.parameters(), lr=self.learning_rate)  # type: ignore
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self.device = device
        self.module.to(device)

    def reset_parameters(self) -> None:
        """ Resets parameters of layers """
        for layer in self.module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()  # type: ignore

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
        x, y = self._x_y_to_tensor(x, y)
        tensor_dataset = TensorDataset(x, y)
        train_dataset, validation_dataset = random_split(tensor_dataset, [1 - self.validation_fraction, self.validation_fraction])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self._train_loop(train_loader, validation_loader, self.epochs)
        return self

    def _x_to_tensor(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        x = x.to(self.device)
        return x

    def _x_y_to_tensor(
        self,
        x: Tensor | np.ndarray | pd.DataFrame,
        y: Tensor | np.ndarray | pd.DataFrame | pd.Series

    ) -> tuple[Tensor, Tensor]:
        x = self._x_to_tensor(x)
        if isinstance(y, np.ndarray):
            if y.dtype.kind == "i":
                y = torch.tensor(y, dtype=torch.int64)
            else:
                y = torch.tensor(y, dtype=torch.float32)
        elif isinstance(y, pd.DataFrame):
            y = torch.tensor(y.values, dtype=torch.float32)
        elif isinstance(y, pd.Series):
            if y.dtype == np.dtype('int64'):
                y = torch.tensor(y.values, dtype=torch.int64)
            else:
                y = torch.tensor(y.values, dtype=torch.float32)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y tensors should have the same first dimension")
        y = y.to(self.device)
        return x, y

    def fit_predict(self, x: Tensor | np.ndarray | pd.DataFrame, y: Tensor | np.ndarray | pd.DataFrame | pd.Series) -> Tensor:
        """ scikit-learn like fit_predict method
        trains the neural network based on training set (x,y) and predicts values for training examples x
        combines fit and predict methods into one.

        Parameters
        ----------
        x: Tensor | np.ndarray | pd.DataFrame
            The training input samples of shape (n_samples, n_features).
        y: Tensor | np.array | pd.DataFrame | pd.Series
            The training target values of shape (n_samples,) or (n_samples, n_outputs).

        Returns
        -------
        y_pred: Tensor
            The predicted values for the training examples x.
        """
        self.fit(x, y)
        return self.predict(x)

    def _train_loop(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int):
        pbar = tqdm(range(epochs), total=epochs, unit="epochs")
        for epoch in pbar:

            self.module.train()
            self._train_one_epoch(train_loader)

            self.module.eval()
            with torch.inference_mode():
                valid_loss = self._validate_one_epoch(validation_loader)
            pbar.set_postfix(loss=valid_loss, epoch=epoch + 1)

    def _train_one_epoch(self, train_loader: DataLoader) -> np.floating:
        losses = []
        pbar = tqdm(train_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            outputs = self.module(x)
            loss = self.loss(outputs, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), batch=batch + 1)

        return np.mean(losses)

    def _validate_one_epoch(self, validation_loader: DataLoader) -> np.floating:
        losses = []
        pbar = tqdm(validation_loader, unit="batches", leave=False)

        for batch, (x, y) in enumerate(pbar):
            outputs = self.module(x)
            loss = self.loss(outputs, y)
            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), batch=batch + 1)

        return np.mean(losses)

    def predict(self, x: Tensor | np.ndarray | pd.DataFrame) -> Tensor:
        """ scikit-learn like predict method
        predicts values for examples input examples x.

        Parameters
        ----------
        x: Tensor | np.ndarray | pd.DataFrame
           The input samples of shape (n_samples, n_features).

        Returns
        -------
        y_pred: Tensor | np.ndarray | pd.DataFrame
            The predicted values for examples x.

        """
        x = self._x_to_tensor(x)
        self.module.eval()
        with torch.inference_mode():
            result = self.module(x).cpu()
        return result

    def set_params(self, **params):
        """ scikit-learn like param setting method
        allows changing parameters of the model set in constructor.

        Parameters
        ----------
        **params: dict
            Keyword arguments representing the parameters to be set.
        """

        def _update_optimizer():

            if self.learning_rate == "auto":
                self.optimizer = self.optimizer_type(self.module.parameters())  # type: ignore
            else:
                self.optimizer_type(self.module.parameters(), lr=self.learning_rate)  # type: ignore

        if not params:
            return self
        valid_params = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {valid_params.keys()!r}."
                )
            if key == "device":
                self.device = value
                self.module.to(self.device)
            elif key == "optimizer_type":
                self.optimizer_type = value
                _update_optimizer()
            elif key == "learning_rate":
                self.learning_rate = value
                _update_optimizer()
            elif key == "module":
                self.module = value
                _update_optimizer()
        return self

    def to_torch_module(self) -> nn.Module:
        """Returns QModel's module with torch neural network.

        Returns:
            nn.Module: Torch neural network.
        """
        return self.module
