""" Base tests for QModel """
import pickle
import pytest
import torch
from torch import nn
from torch import optim
from torch.optim import Adam
import numpy as np
from sklearn.datasets import load_iris

from qailab.torch.qmodel import QModel


def test_runtime():
    """ Runtime test """
    q_model = QModel(torch.nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    ), nn.MSELoss(), optimizer_type=Adam, batch_size=2, epochs=3)
    assert isinstance(q_model, QModel)
    q_model = q_model.fit(np.random.rand(100, 10), np.random.rand(100, 1))
    assert isinstance(q_model, QModel)
    assert len(q_model.predict(np.random.rand(10, 10)) == 10)
    assert len(q_model.fit_predict(np.random.rand(100, 10), np.random.rand(100, 1))) == 100


def test_input_validation():
    """ Input validation test """
    q_model = QModel(torch.nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    ), nn.MSELoss(), optimizer_type=Adam, batch_size=2, epochs=3)
    assert isinstance(q_model, QModel)
    with pytest.raises(ValueError):
        q_model.fit(np.random.rand(200, 10), np.random.rand(100, 1))


def test_changing_params():
    """ Changing params test """
    q_model = QModel(torch.nn.Sequential(nn.Linear(10, 1)), loss=nn.MSELoss())
    q_model = q_model.fit(np.random.rand(100, 10), np.random.rand(100, 1))
    assert len(q_model.predict(np.random.rand(10, 10)) == 10)
    q_model.set_params(module=nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU()), optimizer_type=optim.Adagrad, learning_rate=0.01)
    assert isinstance(q_model.get_params()['module'][0], nn.Conv2d) and isinstance(q_model.get_params()['module'][1], nn.ReLU)
    assert q_model.get_params()['optimizer_type'] == optim.Adagrad
    assert q_model.get_params()['learning_rate'] == 0.01
    q_model = q_model.fit(np.random.rand(100, 3, 100, 100), np.random.rand(100, 1, 98, 98))
    assert len(q_model.predict(np.random.rand(10, 3, 100, 100)) == 10)


def test_pandas_input():
    """ Passing pandas input test """
    x, y = load_iris(return_X_y=True, as_frame=True)
    q_model = QModel(torch.nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
        nn.Softmax()
    ), epochs=2, optimizer_type="adamw", loss=nn.CrossEntropyLoss())
    assert len(torch.argmax(q_model.fit_predict(x, y), dim=1)) == len(y)  # type: ignore


def test_if_picklable():
    """ Test if QModel can be pickled """
    q_model = QModel(torch.nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    ), nn.MSELoss(), optimizer_type=Adam, batch_size=2, epochs=3)
    q_model.fit(np.random.rand(10, 10), np.random.rand(10, 1))
    model_in_str = pickle.dumps(q_model)
    assert isinstance(model_in_str, bytes)
    new_model = pickle.loads(model_in_str)
    assert isinstance(new_model, QModel)
    result = new_model.predict(torch.Tensor([0, 1] * 5))
    assert isinstance(result, torch.Tensor)


def test_if_picklable_after_training():
    """ Test if QModel can be pickled after training """
    q_model = QModel(torch.nn.Sequential(nn.Linear(10, 1)), loss=nn.MSELoss())
    q_model.fit(np.random.rand(10, 10), np.random.rand(10, 1))
    model_in_str = pickle.dumps(q_model)
    assert isinstance(model_in_str, bytes)
    new_model = pickle.loads(model_in_str)
    assert isinstance(new_model, QModel)
    result = new_model.predict(torch.Tensor([0, 1] * 5))
    assert isinstance(result, torch.Tensor)
