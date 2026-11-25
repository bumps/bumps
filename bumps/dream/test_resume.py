"""Checkpoint continuation must not discard learned proposal statistics."""

from copy import deepcopy

import h5py
import numpy as np
import pytest

from .core import Dream
from .crossover import AdaptiveCrossover, Crossover
from .state import h5dump, h5load, save_state, load_state


class Gaussian:
    labels = ["a", "b", "c", "d"]
    bounds = None

    def map(self, x):
        return -0.5 * np.sum(x**2 * [1, 2, 4, 8], axis=1)


@pytest.mark.parametrize("format", ["memory", "hdf5", "export"])
@pytest.mark.parametrize("adaptive", [True, False])
def test_crossover_resume(tmp_path, format, adaptive):
    # Inspect learned weights/statistics without requiring the same trajectory:
    # persisting RNG streams is another part of #367.
    np.random.seed(13)
    population = np.random.RandomState(13).randn(1, 20, 4)
    crossover = AdaptiveCrossover(3) if adaptive else Crossover([0.2, 0.8], [1, 3])
    sampler = Dream(
        model=Gaussian(),
        population=population,
        draws=400,
        burn=400,
        CR=crossover,
        alpha=0,
        monitor=lambda *args: None,
    )
    state = sampler.sample()
    if adaptive:
        assert np.any(sampler.CR._count > 0)
        assert not np.allclose(sampler.CR.weight, 1 / 3)
    reference_crossover = deepcopy(sampler.CR)
    if format == "hdf5":
        with h5py.File(tmp_path / "state.h5", "w") as f:
            h5dump(f, state)
        with h5py.File(tmp_path / "state.h5", "r") as f:
            restored = h5load(f)
    elif format == "export":
        save_state(state, str(tmp_path / "state"))
        restored = load_state(str(tmp_path / "state"))
    else:
        restored = deepcopy(state)

    # The first resumed step must use the learned grid and weights. An immediate
    # abort keeps this independent of the optional C/Numba RNG implementations.
    resumed = Dream(
        model=Gaussian(),
        population=population,
        draws=1200,
        burn=1200,
        alpha=0,
        monitor=lambda *args: None,
    )
    resumed.sample(state=restored, abort_test=lambda: True)
    np.testing.assert_array_equal(resumed.CR.CR, reference_crossover.CR)
    np.testing.assert_array_equal(resumed.CR.weight, reference_crossover.weight)
    if adaptive:
        # One new generation has been accumulated before the abort.
        assert resumed.CR._generations == reference_crossover._generations + 1
        assert resumed.CR._Nchains == reference_crossover._Nchains + 20
        assert np.all(resumed.CR._count >= reference_crossover._count)


@pytest.mark.parametrize("format", ["hdf5", "export"])
def test_legacy_crossover_resume(tmp_path, format):
    population = np.random.RandomState(17).randn(1, 20, 4)
    state = Dream(
        model=Gaussian(),
        population=population,
        draws=400,
        alpha=0,
        monitor=lambda *args: None,
    ).sample()
    # Simulate a pre-crossover-state session without changing its other fields.
    if format == "hdf5":
        with h5py.File(tmp_path / "legacy.h5", "w") as f:
            h5dump(f, state)
            del f["crossover_state"]
        with h5py.File(tmp_path / "legacy.h5", "r") as f:
            restored = h5load(f)
    else:
        from gzip import open as gzopen

        save_state(state, str(tmp_path / "legacy"))
        path = tmp_path / "legacy-stats.mc.gz"
        with gzopen(path, "rt") as f:
            lines = [line for line in f if not line.startswith("# crossover ")]
        with gzopen(path, "wt") as f:
            f.writelines(lines)
        restored = load_state(str(tmp_path / "legacy"))
    resumed = Dream(
        model=Gaussian(),
        population=population,
        draws=800,
        alpha=0,
        monitor=lambda *args: None,
    )
    resumed.sample(state=restored, abort_test=lambda: True)
    np.testing.assert_array_equal(resumed.CR.weight, np.full(3, 1 / 3))
