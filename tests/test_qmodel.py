""" Base tests for QModel """
import pytest
import torch
from torch import nn
from torch import optim
from torch.optim import Adam
import numpy as np
from sklearn.datasets import load_iris

from qlearning.torch.qmodel import QModel


def test_runtime():
    """ Runtime test """
    q_model = QModel([
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    ], Adam, loss=nn.MSELoss(), batch_size=2, epochs=3)
    assert isinstance(q_model, QModel)
    q_model = q_model.fit(np.random.rand(100, 10), np.random.rand(100, 1))
    assert isinstance(q_model, QModel)
    assert len(q_model.predict(np.random.rand(10, 10)) == 10)
    assert len(q_model.fit_predict(np.random.rand(100, 10), np.random.rand(100, 1))) == 100


def test_input_validation():
    """ Input validation test """
    q_model = QModel([
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    ], Adam, loss=nn.MSELoss(), batch_size=2, epochs=3)
    assert isinstance(q_model, QModel)
    with pytest.raises(ValueError):
        q_model.fit(np.random.rand(200, 10), np.random.rand(100, 1))


def test_changing_params():
    """ Changing params test """
    q_model = QModel([nn.Linear(10, 1)])
    q_model = q_model.fit(np.random.rand(100, 10), np.random.rand(100, 1))
    assert len(q_model.predict(np.random.rand(10, 10)) == 10)
    q_model.set_params(layers=[nn.Conv2d(3, 1, 3), nn.ReLU()], optimizer_type=optim.Adagrad, learning_rate=0.01)
    assert isinstance(q_model.get_params()['layers'][0], nn.Conv2d) and isinstance(q_model.get_params()['layers'][1], nn.ReLU)
    assert q_model.get_params()['optimizer_type'] == optim.Adagrad
    assert q_model.get_params()['learning_rate'] == 0.01
    q_model = q_model.fit(np.random.rand(100, 3, 100, 100), np.random.rand(100, 1, 98, 98))
    assert len(q_model.predict(np.random.rand(10, 3, 100, 100)) == 10)


def test_pandas_input():
    """ Passing pandas input test """
    x, y = load_iris(return_X_y=True, as_frame=True)
    q_model = QModel([
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
        nn.Softmax()

    ], epochs=100, optimizer_type="adamw", loss=nn.CrossEntropyLoss())
    assert len(torch.argmax(q_model.fit_predict(x, y), dim=1)) == len(y) # type: ignore
