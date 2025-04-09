from qiskit.primitives import BaseSamplerV2, BaseEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2

from qlearning.qlauncher import AerBackend


def test_Aer_backend_local():
    backend = AerBackend('local_simulator')
    assert isinstance(backend.sampler, BaseSamplerV2)
    assert isinstance(backend.estimator, BaseEstimatorV2)


def test_Aer_backend_backendv1v2():
    backend = AerBackend('backendv1v2_simulator', backendv1v2=FakeAlmadenV2())
    assert isinstance(backend.sampler, BaseSamplerV2)
    assert isinstance(backend.estimator, BaseEstimatorV2)

    assert isinstance(backend.backendv1v2, FakeAlmadenV2)
