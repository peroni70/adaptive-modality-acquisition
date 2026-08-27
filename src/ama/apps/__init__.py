"""Built-in applications.

Importing this package registers every application shipped with ``ama``.
"""

from . import patch_mnist, toy_sensors  # noqa: F401

__all__ = ["patch_mnist", "toy_sensors"]
