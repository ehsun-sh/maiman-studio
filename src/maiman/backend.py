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

**CuPy is not exercised on an ordinary runner.** GitHub's hosted machines have no
CUDA device, so the default CI job tests the part that would actually break a
port: that the kernels never touch NumPy directly, and that a second,
deliberately hostile array library gets identical answers out of them.

Where a device *does* exist, :func:`check_device` runs the real thing — a split
step on CuPy against the same span on NumPy — and the workflow has a job that
calls it on any runner labelled ``gpu``. That job is not scheduled when no such
runner is attached, which is the honest arrangement: it says nothing today
rather than claiming a coverage that is not there, and it starts saying
something the day somebody attaches a machine. ``maiman devices`` is the same
check by hand.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BackendCheck:
    """What happened when one back-end was asked to propagate the same span."""

    name: str
    ran: bool
    max_difference: float | None = None
    agrees: bool = False
    why: str = ""
    """Why it could not run, when it could not — a card present but unusable."""


@dataclass(frozen=True)
class DeviceReport:
    """Which back-ends exist here, and whether the ones that do agree with NumPy."""

    backends: dict[str, bool]
    checks: tuple[BackendCheck, ...]

    @property
    def agrees(self) -> bool:
        """True when every back-end that ran got the same answer.

        A machine with no device at all returns True, because "only NumPy here"
        is an answer rather than a failure and ``maiman devices`` must not exit
        non-zero on every laptop that runs it.
        """
        return all(check.ran and check.agrees for check in self.checks)


def check_device(tolerance: float = 1e-9) -> DeviceReport:
    """Run one propagation on every available back-end and compare the answers.

    The end-to-end check that :mod:`tests.hostile_backend` cannot be: that a
    *real* second array library, on real hardware, gets the same field out of the
    same span. The hostile module proves the kernels never reach for NumPy, which
    is what would break a port; this proves the port is not broken.

    Deliberately a function rather than a test, because the machine that can run
    it is usually not the machine running the suite. ``maiman devices`` calls it,
    so anyone with a card can verify their own install in one command, and the
    workflow's GPU job calls the same thing.

    A soliton, because it is the one propagation whose *correct* answer is known
    without reference to another run: an N=1 soliton comes out of a span with the
    envelope it went in with. Two libraries agreeing on a wrong answer is a real
    failure mode, and this one cannot hide in it.
    """
    import numpy as _np

    from .kernels import propagate_ssfm

    def span(xp: ModuleType) -> object:
        time = xp.linspace(-20.0, 20.0, 4096)
        field = (1.0 / xp.cosh(time)).astype(xp.complex128)
        out, _ = propagate_ssfm(
            field,
            sample_rate=4096.0 / 40.0,
            beta2=-1.0,
            gamma=1.0,
            alpha=0.0,
            distance=1.0,
            max_nonlinear_phase=0.002,
        )
        return out

    reference = to_numpy(span(_np))
    checks: list[BackendCheck] = []
    for name, present in available().items():
        if name == "numpy" or not present:
            continue
        try:
            module = importlib.import_module(name)
            result = to_numpy(span(module))
        except Exception as error:  # a card that is present but unusable
            checks.append(
                BackendCheck(name=name, ran=False, why=f"{type(error).__name__}: {error}")
            )
            continue
        difference = float(_np.abs(result - reference).max())
        checks.append(
            BackendCheck(
                name=name,
                ran=True,
                max_difference=difference,
                agrees=difference <= tolerance,
            )
        )
    return DeviceReport(backends=available(), checks=tuple(checks))
