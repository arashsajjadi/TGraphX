import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (>5s each)")
    config.addinivalue_line("markers", "stress: marks tests as stress/long-running")
