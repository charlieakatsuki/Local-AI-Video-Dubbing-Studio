"""Basic package import test."""

import local_dubbing


def test_package_can_be_imported() -> None:
    """The project package is available to consumers."""
    assert local_dubbing.AppConfig is not None
