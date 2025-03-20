""" Base tests for QModel """
from torch.nn import Linear, ReLU, MSELoss
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
