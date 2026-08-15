def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: downloads multi-GB HF checkpoints and runs full inference",
    )
    config.addinivalue_line(
        "markers",
        "gpu: needs CUDA and flash-attn (the paged attention path)",
    )
