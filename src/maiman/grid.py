"""The ITU channel grids, and why the two of them are not the same kind of grid.

Everything in this library that takes a wavelength has taken any wavelength, and
a project that wanted to sit on a standard grid worked the channel arithmetic out
for itself. That arithmetic is small and it is also exactly the sort of thing
that goes subtly wrong: an anchor half a channel off, or a wavelength grid built
by dividing where it should have been built by multiplying.

**DWDM is uniform in frequency and CWDM is uniform in wavelength**, and that is
the whole difference between them.

ITU-T G.694.1 anchors the dense grid at **193.1 THz** and steps it in exact
multiples of the spacing. Frequency, not wavelength, because what a system has to
keep constant is the guard band between channels as seen by the *filters* — and a
filter's width is a bandwidth. The consequence is that the channels are not
evenly spaced in wavelength: at 100 GHz they sit 0.781 nm apart near 1530 nm and
0.819 nm apart near 1565, because ``dlambda = lambda**2 df / c`` and lambda is
changing across the band. Anyone who builds a "100 GHz grid" by stepping 0.8 nm
gets something that drifts off ITU by most of a channel over the C band.

ITU-T G.694.2 does the opposite: eighteen channels from **1271 nm to 1611 nm, 20
nm apart in wavelength**, which come out unevenly spaced in frequency. The 20 nm
is not arbitrary and it is the point of the standard. A CWDM channel is meant to
be served by an **uncooled** DFB laser, and an uncooled laser's wavelength walks
with its temperature at about 0.1 nm/K: over a 0 to 70 C case-temperature range
that is 7 nm of drift, and adding manufacturing spread on top gives a window
about 13 nm wide that the channel has to contain. 20 nm spacing is what makes a
transmitter with no thermoelectric cooler, no wavelength locker and no control
loop a legal channel — which is why CWDM optics cost a fraction of DWDM optics
and why the grid reaches out to 1611 nm where no amplifier will help it.
"""

from __future__ import annotations

from .units import C_LIGHT, frequency_to_wavelength, wavelength_to_frequency

#: The frequency ITU-T G.694.1 anchors the dense grid to [Hz]. Every DWDM channel
#: at every spacing is an exact multiple of the spacing away from this, which is
#: what makes a 50 GHz plan a superset of a 100 GHz one rather than something
#: offset from it by half a channel.
DWDM_ANCHOR = 193.1e12

#: Channel spacings G.694.1 defines [Hz]. 200 GHz is in the standard as a
#: multiple rather than a grid of its own, and is here because legacy systems are
#: full of it.
DWDM_SPACINGS = (12.5e9, 25e9, 50e9, 100e9, 200e9)

#: The first CWDM channel of G.694.2 [m], and the step between them. Eighteen
#: channels, 1271 nm to 1611 nm.
CWDM_FIRST = 1271e-9
CWDM_SPACING = 20e-9
CWDM_CHANNELS = 18


def dwdm_frequencies(
    count: int, *, spacing: float = 100e9, first: int = 0, anchor: float = DWDM_ANCHOR
) -> tuple[float, ...]:
    """``count`` channels of the G.694.1 dense grid [Hz].

    ``first`` is the channel index relative to the anchor and may be negative, so
    the C band at 100 GHz is roughly ``first=-20`` through ``first=30``. Indices
    rather than the various vendor channel-number conventions, because those
    disagree with each other and the standard does not define one: what G.694.1
    defines is the anchor and the arithmetic.

    ``spacing`` is checked against the standard rather than taken on trust. A
    grid at some other spacing is a perfectly reasonable thing to simulate and
    :func:`channel_frequencies` will build one; it is simply not an ITU grid, and
    a function named after a standard should not hand one back as though it were.
    """
    if spacing not in DWDM_SPACINGS:
        raise ValueError(
            f"{spacing / 1e9:g} GHz is not a G.694.1 spacing; the standard defines "
            f"{[s / 1e9 for s in DWDM_SPACINGS]} GHz. For an arbitrary grid use "
            f"channel_frequencies(), which makes no claim about a standard."
        )
    return channel_frequencies(count, spacing=spacing, first=first, anchor=anchor)


def channel_frequencies(
    count: int, *, spacing: float, first: int = 0, anchor: float = DWDM_ANCHOR
) -> tuple[float, ...]:
    """``count`` equally spaced frequencies [Hz], on any grid at all.

    The arithmetic behind :func:`dwdm_frequencies`, without the claim that the
    result is a standard. Useful for a plan somebody is still choosing the
    spacing of, and for the tests that compare a grid against the band it has to
    fit in.
    """
    if count < 1:
        raise ValueError(f"a grid needs at least one channel, got {count}")
    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    frequencies = tuple(anchor + (first + index) * spacing for index in range(count))
    if frequencies[0] <= 0.0:
        raise ValueError(
            f"channel {first} sits at {frequencies[0] / 1e12:g} THz, which is not a "
            f"frequency; first is an index relative to the {anchor / 1e12:g} THz anchor "
            f"and not a frequency itself"
        )
    return frequencies


def dwdm_wavelengths(
    count: int, *, spacing: float = 100e9, first: int = 0, anchor: float = DWDM_ANCHOR
) -> tuple[float, ...]:
    """The same channels as wavelengths [m], in the order the frequencies are.

    **Descending**, because wavelength runs backwards against frequency, and the
    spacing between neighbours is not constant — which is the thing about a
    frequency grid that catches people out. See the module docstring.
    """
    return tuple(
        frequency_to_wavelength(f)
        for f in dwdm_frequencies(count, spacing=spacing, first=first, anchor=anchor)
    )


def cwdm_wavelengths(count: int = CWDM_CHANNELS, *, first: int = 0) -> tuple[float, ...]:
    """``count`` channels of the G.694.2 coarse grid [m], from 1271 nm in 20 nm steps.

    ``first`` is the index of the first channel wanted, 0 being 1271 nm. The
    standard defines eighteen and stops, so asking for more is refused rather
    than extrapolated: 1631 nm is not a CWDM channel, it is past the end of the
    grid, and continuing the arithmetic would invent one.
    """
    if count < 1:
        raise ValueError(f"a grid needs at least one channel, got {count}")
    if first < 0:
        raise ValueError(f"the coarse grid starts at channel 0 (1271 nm), got first={first}")
    if first + count > CWDM_CHANNELS:
        raise ValueError(
            f"G.694.2 defines {CWDM_CHANNELS} channels, 1271 nm to 1611 nm; "
            f"first={first} and count={count} runs to channel {first + count - 1}"
        )
    return tuple(CWDM_FIRST + (first + index) * CWDM_SPACING for index in range(count))


def cwdm_frequencies(count: int = CWDM_CHANNELS, *, first: int = 0) -> tuple[float, ...]:
    """The coarse channels as frequencies [Hz]. Unevenly spaced, being a wavelength grid."""
    return tuple(wavelength_to_frequency(w) for w in cwdm_wavelengths(count, first=first))


def wavelength_spacing(frequency: float, spacing: float) -> float:
    """Wavelength between two channels ``spacing`` apart at ``frequency`` [m].

    ``lambda**2 * df / c``, the conversion that makes a frequency grid uneven in
    wavelength: 0.800 nm for 100 GHz at 1550 nm, and 0.781 at 1530.
    """
    if frequency <= 0.0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    return frequency_to_wavelength(frequency) ** 2 * spacing / C_LIGHT
