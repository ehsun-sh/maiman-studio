"""The PCS mapper, and the mutual information the analyser credits it with.

A shaped source sends some points more often than others, so it cannot be
judged by its bit error rate: what a receiver can be credited with is the
mutual information between what was sent and what arrived. Both halves are
tested here: that the mapper sends the distribution it claims, and that the
analyser's estimate of the information is the one the textbook defines.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.analysis import measure_constellation
from maiman.components import PCSMapper, PRBSGenerator
from maiman.components.mapping import maxwell_boltzmann, quantised
from maiman.modulation import qam_constellation
from maiman.signals import SymbolSignal


def entropy_of(p: np.ndarray) -> float:
    used = p[p > 0]
    return float(-np.sum(used * np.log2(used)))


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("target", [2.5, 3.28, 3.9])
def test_maxwell_boltzmann_hits_the_entropy_it_is_asked_for(bits: int, target: float) -> None:
    points = qam_constellation(bits)
    p = maxwell_boltzmann(points, target)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert entropy_of(p) == pytest.approx(target, abs=1e-6)
    # Rarer the further out, and equal on a ring.
    order = np.argsort(np.abs(points))
    assert np.all(np.diff(p[order]) <= 1e-12)


def test_full_entropy_is_the_uniform_alphabet() -> None:
    p = maxwell_boltzmann(qam_constellation(4), 4.0)
    assert np.allclose(p, 1 / 16)


def test_quantised_counts_sum_to_the_steps_and_none_is_zero() -> None:
    p = maxwell_boltzmann(qam_constellation(4), 3.28)
    counts = quantised(p, 256)
    assert counts.sum() == 256
    assert counts.min() >= 1
    assert np.all(np.abs(counts / 256 - p) <= 1 / 256 + 1e-12)


def test_the_constellation_has_unit_mean_power_under_its_own_distribution() -> None:
    mapper = PCSMapper(alphabet=16.0, entropy=3.28)
    power = np.sum(mapper.probabilities() * np.abs(mapper.constellation()) ** 2)
    assert power == pytest.approx(1.0, abs=1e-12)


def test_the_mapper_sends_the_distribution_it_claims() -> None:
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=2, sequence_length=1 << 15, seed=1)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=8.0, label="prbs"))
    mapper = graph.add(PCSMapper(alphabet=16.0, entropy=3.5, label="map"))
    graph.connect(prbs, mapper["in"])
    sent = graph.run(keep=[mapper]).port(mapper, "out")
    assert isinstance(sent, SymbolSignal)
    points = np.asarray(sent.constellation)
    index = np.argmin(np.abs(np.asarray(sent.symbols)[:, None] - points[None, :]), axis=1)
    seen = np.bincount(index, minlength=16) / index.size
    assert np.allclose(seen, mapper.probabilities(), atol=0.01)
    assert entropy_of(seen) == pytest.approx(3.5, abs=0.02)


def test_the_mapper_refuses_a_source_that_is_not_eight_bits_a_symbol() -> None:
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=2, sequence_length=256, seed=1)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=4.0, label="prbs"))
    mapper = graph.add(PCSMapper(label="map"))
    graph.connect(prbs, mapper["in"])
    with pytest.raises(ValueError, match="bits per symbol to 8"):
        graph.run()


@pytest.mark.parametrize("entropy", [1.0, 4.5])
def test_the_mapper_refuses_an_entropy_it_cannot_have(entropy: float) -> None:
    with pytest.raises(ValueError, match="entropy"):
        PCSMapper(alphabet=16.0, entropy=entropy).validate()


def awgn(snr_db: float, p: np.ndarray, n: int = 1 << 16) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(3)
    points = qam_constellation(4)
    points = points / math.sqrt(float(np.sum(p * np.abs(points) ** 2)))
    sent = points[rng.choice(points.size, size=n, p=p)]
    sigma = math.sqrt(10 ** (-snr_db / 10) / 2)
    noise = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return sent + noise, sent, points


def test_mutual_information_of_uniform_16qam_on_a_gaussian_channel() -> None:
    """The textbook curve: 3.18 bit at 10 dB, 3.92 at 15 dB, 4 with no noise."""
    uniform = np.full(16, 1 / 16)
    for snr_db, expected in ((10.0, 3.18), (15.0, 3.92)):
        received, sent, points = awgn(snr_db, uniform)
        result = measure_constellation(received, sent, points)
        assert result.entropy == pytest.approx(4.0, abs=0.01)
        assert result.mutual_information == pytest.approx(expected, abs=0.02)
    received, sent, points = awgn(60.0, uniform)
    assert measure_constellation(received, sent, points).mutual_information == pytest.approx(
        4.0, abs=1e-3
    )


def test_shaping_gains_at_low_snr_and_never_beats_its_entropy() -> None:
    uniform = np.full(16, 1 / 16)
    shaped = maxwell_boltzmann(qam_constellation(4), 3.7)
    base = measure_constellation(*awgn(9.0, uniform)).mutual_information
    gain = measure_constellation(*awgn(9.0, shaped)).mutual_information
    assert gain > base + 0.05
    clean = measure_constellation(*awgn(40.0, shaped))
    assert clean.mutual_information <= clean.entropy
    assert clean.entropy == pytest.approx(3.7, abs=0.02)
