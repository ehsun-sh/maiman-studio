"""Which array library a kernel runs on, and how it finds out.

The propagation kernels are the only part of this project a GPU would help: they
are a loop over FFTs on a long array, and everything else is either scalar
arithmetic or a closed form evaluated a few dozen times. :mod:`maiman.kernels`
was written as array-to-array functions from the beginning so that this could be
added without touching anything above it, and this module is the whole of the
addition.

**The arrays decide, not a setting.** A kernel handed CuPy arrays runs on CuPy
and returns CuPy arrays; handed NumPy arrays it runs on NumPy. There is no global
mode, no flag on the context, and nothing to get out of step between two calls —
which matters because the kernels are pure functions and a hidden mode would be
the one piece of state that could make the same inputs give different answers.
It is also the idiom CuPy itself recommends.

Dispatch is by the array's own type: ``type(a).__module__`` names the package it
came from, and if that package is imported and looks like an array library it is
the one to use. No registry, nothing to keep in sync, and a library nobody has
heard of works if it presents the same surface.

**What a back-end has to provide** is small, and
``tests/test_backend.py`` holds it to exactly that list — a kernel that reaches
past it for something only NumPy has fails there rather than on somebody's GPU.

**CuPy is not exercised here.** No CUDA device or CuPy install is available in
this project's CI, so what is tested is that the kernels never touch NumPy
directly and that a second, deliberately hostile array library gets identical
answers out of them. That is the part that would break a port; the part that
remains untested is CuPy's own numerics, which are not this project's to test.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

#: Attributes a module must have before it is believed to be an array library.
#: ``fft`` is the one every kernel here needs and no ordinary module has.
_MARKERS = ("fft", "exp", "abs")


def array_module(*arrays: object) -> ModuleType:
    """The array library ``arrays`` belong to, defaulting to NumPy.

    The first argument that names a plausible array library wins, so a kernel
    mixing a device array with a plain Python float still runs on the device.
    Scalars, lists and NumPy arrays all fall through to NumPy, which is what
    makes this free for every existing caller.
    """
    for array in arrays:
        root = type(array).__module__.partition(".")[0]
        module = sys.modules.get(root)
        if module is None or module is np:
            continue
        if all(hasattr(module, marker) for marker in _MARKERS):
            return module
    return np


def to_numpy(array: object) -> np.ndarray:
    """Bring a result back to the host, whatever it was computed on.

    Measurement and encoding are host work — a constellation histogram is
    kilobytes and a JSON payload is not going anywhere near a device — so this is
    the boundary the rest of the project sees.
    """
    getter = getattr(array, "get", None)
    if callable(getter):  # cupy.ndarray.get, and anything that copies it
        return np.asarray(getter())
    return np.asarray(array)


def available() -> dict[str, bool]:
    """Which back-ends this interpreter could actually use.

    Reported rather than assumed: ``pip install cupy-cuda12x`` and a device is
    what makes the second entry true, and neither is something this project can
    check for by importing hopefully.
    """
    import importlib.util

    return {
        "numpy": True,
        "cupy": importlib.util.find_spec("cupy") is not None,
    }
