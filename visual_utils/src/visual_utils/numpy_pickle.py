"""Read experiment exports containing primitive metadata and NumPy arrays."""

from __future__ import annotations

import importlib
import pickle

import numpy as np


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Limit pickle globals to NumPy array and scalar reconstruction."""

    error_type = pickle.UnpicklingError

    def find_class(self, module: str, name: str):
        if (module, name) in {("numpy", "ndarray"), ("numpy", "dtype")}:
            return getattr(np, name)
        core = "numpy._core" if hasattr(np, "_core") else "numpy.core"
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            return getattr(importlib.import_module(core + ".multiarray"), name)
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            return getattr(importlib.import_module(core + ".numeric"), name)
        raise self.error_type(
            f"Unsupported pickle global {module}.{name}. Export NumPy arrays and primitive metadata; "
            "Torch tensors and model objects are not accepted."
        )
