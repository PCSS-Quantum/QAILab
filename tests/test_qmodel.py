""" Base tests for QModel """
from torch.nn import Linear, ReLU, MSELoss, Conv2d
from torch import optim
from torch.optim import Adam
import numpy as np

from qlearning.torch.qmodel import QModel


def test_runtime():
    """ Runtime test """
    q_model = QModel([
        Linear(10, 5),
        ReLU(),
        Linear(5, 1),
    ], Adam, MSELoss(), batch_size=2, epochs=3)
    assert isinstance(q_model, QModel)
    q_model = q_model.fit(np.random.rand(100, 10), np.random.rand(100, 1))
    assert isinstance(q_model, QModel)
    assert len(q_model.predict(np.random.rand(10, 10)) == 10)
    assert len(q_model.fit_predict(np.random.rand(100, 10), np.random.rand(100, 1))) == 100


def test_changing_params():
    """ Testing params test """
    q_model = QModel([Linear(10, 1)])
    assert isinstance(q_model.get_params(), dict)
    q_model = q_model.fit(np.random.rand(100, 10), np.random.rand(100, 1))
    assert len(q_model.predict(np.random.rand(10, 10)) == 10)
    q_model.set_params(layers=[Conv2d(3, 1, 3), ReLU()], optimizer_type=optim.Adagrad)
    assert isinstance(q_model.get_params(), dict)
    q_model = q_model.fit(np.random.rand(100, 3, 100, 100), np.random.rand(100, 1, 98, 98))
    assert len(q_model.predict(np.random.rand(10, 3, 100, 100)) == 10)
