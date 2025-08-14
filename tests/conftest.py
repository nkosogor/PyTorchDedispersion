# tests/conftest.py
import os
import torch
import pytest

# Keep CPU determinism and avoid oversubscription stalls on CI
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

def pytest_addoption(parser):
    parser.addoption("--ci", action="store_true", help="Run with small test sizes for CI")

@pytest.fixture(scope="session")
def small_ci(request):
    return request.config.getoption("--ci")

