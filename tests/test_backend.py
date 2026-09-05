"""The array-library indirection the propagation kernels run through.

CuPy is not installed here and there is no device to run it on, so what is
checked is the half that would actually break a port: that the kernels take
their array library from their inputs and never reach past it for NumPy.

:mod:`hostile_backend` is what makes that checkable, and it is calibrated to what
would really break rather than to what looks dangerous. Universal functions are
not the problem: ``np.exp`` on a CuPy array dispatches and comes back a CuPy
array, and so do ``+`` and ``*``. What breaks is anything that *allocates* —
``np.fft.fftfreq`` and ``np.zeros`` build on the host, and a kernel calling one
inside its loop pays a transfer every step. Those raise here.

The kernels are then run on both libraries and the results compared, and the set
of names the second library was asked for is recorded — that set is the contract
a real back-end has to meet.
"""

from __future__ import annotations

import numpy as np
import pytest

import hostile_backend
from maiman.backend import array_module, available, to_numpy
from maiman.kernels import (
    angular_frequency_grid,
    apply_pmd,
    lowpass_filter,
    propagate_coupled_ssfm,
    propagate_dispersion,
    random_pmd_sections,
)

FS = 160e9
SAMPLES = 512
BETA2 = -2.17e-26
BETA3 = 1.3e-40


def pulse() -> np.ndarray:
    times = (np.arange(SAMPLES) - SAMPLES // 2) / FS
    return np.exp(-((times * FS / 40.0) ** 2)) * (1.0 + 0.3j)


# ---------------------------------------------------------------------------
# the dispatch


def test_plain_arrays_and_scalars_land_on_numpy() -> None:
    """Which is what makes this free for every caller that existed before it."""
    assert array_module(np.zeros(3)) is np
    assert array_module(1.0) is np
    assert array_module([1, 2, 3]) is np
    assert array_module() is np
    assert array_module(None, "a string", 2) is np


def test_an_array_from_another_library_takes_the_kernels_with_it() -> None:
    """The array names its own package, and that package is the one used."""
    guarded = hostile_backend.asarray(np.zeros(3))
    assert array_module(guarded) is hostile_backend
    # The first argument that names a library wins, so a device array mixed with
    # a float still runs on the device.
    assert array_module(1.0, guarded) is hostile_backend
    assert array_module(guarded, np.zeros(3)) is hostile_backend


def test_a_module_that_is_not_an_array_library_is_not_mistaken_for_one() -> None:
    """``type(x).__module__`` names a package; most packages are not NumPy-shaped."""

    class Ordinary:
        pass

    assert array_module(Ordinary()) is np


def test_results_come_back_to_the_host_whatever_they_were_computed_on() -> None:
    """And by copying, not by hoping ``np.asarray`` will do it.

    ``np.asarray`` on a real device array raises rather than transferring — CuPy
    refuses to move data across the bus without being asked. So the copy has to
    go through the array's own ``get``, and this checks that it did rather than
    that the answer came out right, because here it would either way.
    """
    hostile_backend.reset()
    guarded = hostile_backend.asarray(np.arange(4.0))
    brought = to_numpy(guarded)

    assert hostile_backend.FETCHED, "to_numpy did not ask the array to copy itself"
    assert type(brought) is np.ndarray
    assert np.array_equal(brought, np.arange(4.0))

    # A host array has no `get`, and must come back untouched.
    hostile_backend.reset()
    assert np.array_equal(to_numpy(np.arange(4.0)), np.arange(4.0))
    assert not hostile_backend.FETCHED


def test_what_is_available_is_reported_not_assumed() -> None:
    surface = available()
    assert surface["numpy"] is True
    assert isinstance(surface["cupy"], bool)


# ---------------------------------------------------------------------------
# the kernels, on both libraries


def test_the_guard_really_does_refuse_numpy() -> None:
    """Otherwise every test below would pass without proving anything.

    This is the assertion the rest of the file rests on: if these arrays could be
    used by NumPy directly, a kernel that bypassed the indirection would get the
    right answer and nothing would notice.
    """
    guarded = hostile_backend.asarray(np.ones(4))
    for allocating in (
        lambda: np.fft.fft(guarded),
        lambda: np.fft.rfft(guarded),
        lambda: np.concatenate([guarded, guarded]),
        lambda: np.zeros_like(guarded),
        lambda: np.max(guarded),
    ):
        with pytest.raises(TypeError):
            allocating()

    # And what a device array genuinely does support is left working, because
    # refusing it would be testing a rule that is not true of CuPy either.
    assert isinstance(np.exp(guarded), hostile_backend.Array)
    assert isinstance(guarded * 2.0, hostile_backend.Array)


def test_the_frequency_grid_is_built_where_it_is_asked_for() -> None:
    host = angular_frequency_grid(SAMPLES, FS)
    guest = angular_frequency_grid(SAMPLES, FS, like=hostile_backend.asarray(np.zeros(1)))
    assert isinstance(guest, hostile_backend.Array)
    assert np.allclose(to_numpy(guest), host, rtol=0.0, atol=0.0)


def test_dispersion_gives_the_same_answer_on_either_library() -> None:
    field = pulse()
    host = propagate_dispersion(field, FS, BETA2, 80e3, BETA3)
    guest = propagate_dispersion(hostile_backend.asarray(field), FS, BETA2, 80e3, BETA3)

    assert isinstance(guest, hostile_backend.Array)
    assert np.allclose(to_numpy(guest), host, rtol=0.0, atol=1e-15 * np.abs(host).max())


def test_the_split_step_solver_runs_on_either_library() -> None:
    """The one that matters: a loop over FFTs, which is what a GPU is for."""
    fields = [pulse(), 0.4 * pulse()]
    common = {
        "beta2": [BETA2, BETA2],
        "walkoff": [0.0, 1e-14],
        "beta3": [BETA3, BETA3],
        "gamma": 1.3e-3,
        "alpha": 4.6e-5,
        "distance": 40e3,
    }
    host, host_diagnostics = propagate_coupled_ssfm(fields, FS, **common)  # type: ignore[arg-type]
    guest, guest_diagnostics = propagate_coupled_ssfm(
        [hostile_backend.asarray(f) for f in fields],
        FS,
        **common,  # type: ignore[arg-type]
    )

    assert guest_diagnostics.steps == host_diagnostics.steps
    assert guest_diagnostics.peak_nonlinear_phase == pytest.approx(
        host_diagnostics.peak_nonlinear_phase, rel=1e-12
    )
    for one, other in zip(host, guest, strict=True):
        assert isinstance(other, hostile_backend.Array)
        assert np.allclose(to_numpy(other), one, rtol=0.0, atol=1e-14 * np.abs(one).max())


def test_polarization_mode_dispersion_runs_on_either_library() -> None:
    sections = random_pmd_sections(2e-12, 8, np.random.default_rng(3))
    ex, ey = pulse(), 0.5 * pulse()
    host = apply_pmd(ex, ey, FS, sections)
    guest = apply_pmd(hostile_backend.asarray(ex), hostile_backend.asarray(ey), FS, sections)
    for one, other in zip(host, guest, strict=True):
        assert isinstance(other, hostile_backend.Array)
        assert np.allclose(to_numpy(other), one, rtol=0.0, atol=1e-14 * np.abs(one).max())


def test_the_electrical_filter_runs_on_either_library() -> None:
    samples = np.real(pulse())
    host = lowpass_filter(samples, FS, 20e9)
    guest = lowpass_filter(hostile_backend.asarray(samples), FS, 20e9)
    assert isinstance(guest, hostile_backend.Array)
    assert np.allclose(to_numpy(guest), host, rtol=0.0, atol=1e-14 * np.abs(host).max())


def test_the_solver_builds_its_frequency_grid_on_the_guest() -> None:
    """Not on the host, and this is the only place that shows.

    A grid built with NumPy still gives the right numbers — multiplying a host
    array by a device array works, by transferring the host one. It works once
    per step, over a PCIe bus, which is the difference between a GPU being worth
    it and not. The allocation is what has to happen in the right place, so the
    allocation is what is asserted.
    """
    hostile_backend.reset()
    propagate_coupled_ssfm(
        [hostile_backend.asarray(pulse())],
        FS,
        beta2=[BETA2],
        walkoff=[0.0],
        gamma=0.0,
        alpha=0.0,
        distance=10e3,
    )
    assert "fft.fftfreq" in hostile_backend.ASKED, sorted(hostile_backend.ASKED)


# ---------------------------------------------------------------------------
# the contract


def test_a_backend_has_to_provide_exactly_this_much() -> None:
    """The surface the kernels use, measured rather than guessed.

    A change that reaches for something only NumPy has fails here, in this
    repository, rather than on somebody's GPU — and a change that stops needing
    something fails here too, which is the point of pinning the set rather than a
    lower bound on it. CuPy provides all of these; that is why it is the back-end
    named in the roadmap.
    """
    hostile_backend.reset()

    fields = [pulse(), 0.4 * pulse()]
    propagate_dispersion(hostile_backend.asarray(pulse()), FS, BETA2, 80e3, BETA3)
    propagate_coupled_ssfm(
        [hostile_backend.asarray(f) for f in fields],
        FS,
        beta2=[BETA2, BETA2],
        walkoff=[0.0, 1e-14],
        beta3=[BETA3, BETA3],
        gamma=1.3e-3,
        alpha=4.6e-5,
        distance=40e3,
    )
    apply_pmd(
        hostile_backend.asarray(pulse()),
        hostile_backend.asarray(pulse()),
        FS,
        random_pmd_sections(2e-12, 4, np.random.default_rng(1)),
    )
    lowpass_filter(hostile_backend.asarray(np.real(pulse())), FS, 20e9)

    assert set(hostile_backend.ASKED) == {
        "abs",
        "complex128",
        "conj",
        "exp",
        "fft.fft",
        "fft.fftfreq",
        "fft.ifft",
        "fft.irfft",
        "fft.rfft",
        "fft.rfftfreq",
        "float64",
        "max",
        "pi",
    }, sorted(hostile_backend.ASKED)


def test_the_kernels_do_not_reach_for_numpy_in_the_hot_path() -> None:
    """Read off the source, because a test can only run the paths it thinks of.

    Only the propagation kernels are converted: the closed forms and the 2x2
    Jones algebra are scalar work a device would slow down, and moving them would
    be cost with no benefit. This checks the ones that are, and names the ones
    that are not so the list is a decision rather than an oversight.
    """
    import inspect

    from maiman import kernels

    converted = (
        kernels.angular_frequency_grid,
        kernels.propagate_dispersion,
        kernels.propagate_coupled_ssfm,
        kernels.apply_pmd,
        kernels.lowpass_filter,
        kernels.gaussian_lowpass_response,
        kernels.super_gaussian_response,
        kernels._total_power,
    )
    for function in converted:
        body = "".join(
            line
            for line in inspect.getsource(function).splitlines(keepends=True)
            if not line.lstrip().startswith("#")
        )
        _, _, after = body.partition('"""')
        _, _, code = after.partition('"""')
        # Host scalars are not the problem and never were: `np.inf` is a float
        # and an annotation is not a call. What breaks a port is allocating.
        allowed = ("np.ndarray", "np.inf", "np.nan")
        offenders = [
            fragment
            for fragment in code.split()
            if fragment.startswith("np.") and not fragment.startswith(allowed)
        ]
        assert not offenders, f"{function.__name__} reaches for {offenders}"
