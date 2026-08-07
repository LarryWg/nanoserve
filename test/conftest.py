import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: downloads multi-GB HF checkpoints and runs full inference",
    )
