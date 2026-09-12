"""Step 3 environment smoke tests."""


def test_package_imports() -> None:
    import rca_sim

    assert rca_sim.__version__ == "0.1.0"
