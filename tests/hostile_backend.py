"""A second array library that refuses to be used by accident.

CuPy cannot be run here — there is no device and no install — so the thing that
would actually break a GPU port is tested instead: that the kernels never reach
for NumPy directly.

The arrays this module hands out are NumPy arrays with ``__array_function__`` set
to ``None``, NumPy's documented way for a type to say it is not NumPy's. That is
calibrated to what would actually break a port rather than to what looks
dangerous. Universal functions are *not* refused, because they are not a problem:
``np.exp`` on a CuPy array dispatches through ``__array_ufunc__`` and comes back
a CuPy array, and so do ``+`` and ``*``, which are ufuncs underneath. What does
break is anything that allocates — ``np.fft.fftfreq``, ``np.zeros``, ``np.eye``
build arrays on the *host*, and a kernel that calls one inside its loop pays a
transfer per step or gets a type error for its trouble. ``np.asarray`` on a
device array is the same problem from the other side.

So those raise here, and a kernel that reaches for one fails rather than quietly
being slow on somebody else's machine.

Every attribute this module is asked for is recorded, so the set of names a
back-end has to provide is a measurement rather than a guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Every attribute the kernels asked of this module, in the order first seen.
ASKED: list[str] = []

#: One entry per call to :meth:`Array.get`, so a host copy is observable.
FETCHED: list[bool] = []


class Array(np.ndarray):
    """A NumPy array that refuses NumPy's *allocating* API.

    Ufuncs are left alone deliberately: ``a * 2`` and ``np.exp(a)`` are the same
    call underneath, both work on a CuPy array, and refusing them would be
    testing a rule that is not true. ``__array_function__ = None`` covers the
    rest — ``np.fft.fft``, ``np.zeros``, ``np.asarray`` and everything else that
    would put the result on the wrong side of a PCIe bus.
    """

    __array_function__ = None  # type: ignore[assignment]

    def get(self) -> np.ndarray:
        """Copy to the host, the way ``cupy.ndarray.get`` does.

        Recorded, because ``maiman.backend.to_numpy`` reaching for ``np.asarray``
        instead would work here — a subclass of ``ndarray`` already is one — and
        raise on a real device array.
        """
        FETCHED.append(True)
        return self.view(np.ndarray)


def wrap(value: Any) -> Any:
    """Put a result back behind the guard, leaving scalars alone."""
    if isinstance(value, np.ndarray):
        return value.view(Array)
    if isinstance(value, tuple):
        return tuple(wrap(item) for item in value)
    return value


def unwrap(value: Any) -> Any:
    """Take the guard off so NumPy will touch it."""
    if isinstance(value, Array):
        return value.view(np.ndarray)
    if isinstance(value, (list, tuple)):
        return type(value)(unwrap(item) for item in value)
    return value


def asarray(value: Any, *args: Any, **kwargs: Any) -> Array:
    return wrap(np.asarray(unwrap(value), *args, **kwargs))


class _Namespace:
    """Forwards to a NumPy namespace, unwrapping in and wrapping out."""

    def __init__(self, source: Any, prefix: str) -> None:
        self._source = source
        self._prefix = prefix

    def __getattr__(self, name: str) -> Any:
        full = f"{self._prefix}{name}"
        if full not in ASKED:
            ASKED.append(full)
        attribute = getattr(self._source, name)
        if not callable(attribute):
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            return wrap(attribute(*unwrap(list(args)), **{k: unwrap(v) for k, v in kwargs.items()}))

        return call


fft = _Namespace(np.fft, "fft.")


def __getattr__(name: str) -> Any:
    """Everything else, forwarded to NumPy and recorded on the way past."""
    if name.startswith("_"):
        raise AttributeError(name)
    if name not in ASKED:
        ASKED.append(name)
    attribute = getattr(np, name)
    if not callable(attribute) or isinstance(attribute, type):
        return attribute  # pi, complex128, float64 and friends

    def call(*args: Any, **kwargs: Any) -> Any:
        return wrap(attribute(*unwrap(list(args)), **{k: unwrap(v) for k, v in kwargs.items()}))

    return call


def reset() -> None:
    ASKED.clear()
    FETCHED.clear()
