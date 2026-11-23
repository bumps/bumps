"""
Samples from a posterior distribution produced by a nested sampler, and the
progress of the sampler as it runs.
"""

__all__ = ["NestedDraw", "NestedProgress", "NestedStats", "kish_ess"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from .bayes import BayesDraw
from .util import NDArray

if TYPE_CHECKING:
    from dynesty import DynamicNestedSampler
    from dynesty.utils import Results


class NestedDraw(BayesDraw):
    """
    A sample from a posterior distribution produced by a nested sampler.

    The points are equally weighted and unordered, so the statistics and
    plots for an ordinary posterior sample apply unchanged. The chain
    history plots do not, and are skipped for anything which is not an
    :class:`~bumps.dream.state.MCMCDraw`.

    *results* is the raw :class:`dynesty.utils.Results` from the sampler,
    which holds the unequally weighted points and the evidence estimate.
    It is copied so that it is not modified by a sampler which keeps
    running, and is None if the sample did not come from dynesty.
    """

    results: Optional["Results"] = None

    def __init__(self, *args, results: Optional["Results"] = None, **kw):
        super().__init__(*args, **kw)
        self.results = results.copy() if results is not None else None


Phase = Literal["initial", "batch", "done"]


@dataclass
class NestedStats:
    """
    A snapshot of a nested sampling run.

    The series hold one value per iteration of the run so far. During the
    batch phase they describe the run as of the last completed batch.
    """

    phase: Phase
    """*initial* run, adding a *batch*, or *done*."""
    batch: int
    """Number of the batch being added, or 0 during the initial run."""
    logz: NDArray
    """ln Z, the log evidence."""
    logzerr: NDArray
    """Standard deviation of ln Z."""
    logvol: NDArray
    """ln X, the log of the prior volume remaining."""
    ess: NDArray
    """Effective sample size of the posterior."""
    information: float
    """Information H, in nats."""
    nlive: int
    """Live points in the current run or batch."""
    iteration: int
    """Iterations taken, including those of the current batch."""
    ncall: int
    """Likelihood evaluations."""
    dlogz_target: float
    """Stopping target for the remaining ln Z in the initial run, and for σ(ln Z) after it."""
    n_effective_target: int
    """Stopping target for the effective sample size."""
    delta_logz: Optional[float] = None
    """ln Z still to be gained from the live points; initial run only."""
    stop_post: Optional[float] = None
    """Posterior stopping value, ``n_effective_target / ess``; batch phase only."""
    stop_evid: Optional[float] = None
    """Evidence stopping value, ``logzerr / dlogz_target``; batch phase only."""
    stop: Optional[float] = None
    """Combination of the stopping values, weighted by *pfrac*; sampling stops at ≤ 1."""


@dataclass
class NestedProgress:
    """
    The state of a running nested sampler, captured by :meth:`snapshot`.

    The fitter updates the fields as the run proceeds. Snapshots read the
    sampler results, so they must be taken on the fitting thread.
    """

    sampler: "DynamicNestedSampler"
    dlogz_target: float
    n_effective_target: int
    phase: Phase = "initial"
    batch: int = 0
    delta_logz: Optional[float] = None
    stop_values: Optional[tuple[float, float, float]] = None
    """``(stop_post, stop_evid, stop)`` from the dynesty stopping function."""

    def snapshot(self) -> NestedStats:
        results = self.sampler.results
        live = self.sampler.batch_sampler or self.sampler.sampler
        stop_post, stop_evid, stop = self.stop_values or (None, None, None)
        return NestedStats(
            phase=self.phase,
            batch=self.batch,
            logz=results.logz,
            logzerr=results.logzerr,
            logvol=results.logvol,
            ess=kish_ess(results.logwt),
            information=float(results.information[-1]),
            nlive=live.nlive,
            iteration=self.sampler.it - 1,
            ncall=self.sampler.ncall,
            dlogz_target=self.dlogz_target,
            n_effective_target=self.n_effective_target,
            delta_logz=self.delta_logz,
            stop_post=stop_post,
            stop_evid=stop_evid,
            stop=stop,
        )


def kish_ess(logwt: NDArray) -> NDArray:
    """
    Return the Kish effective sample size, ``(Σw)² / Σw²``, of each prefix of
    the unnormalized log weights *logwt*.
    """
    logwt = np.asarray(logwt, "d")
    return np.exp(2 * np.logaddexp.accumulate(logwt) - np.logaddexp.accumulate(2 * logwt))


def test_kish_ess():
    from numpy.testing import assert_allclose

    assert_allclose(kish_ess(np.zeros(4)), [1, 2, 3, 4])
    # weights 1, 1, 2: (1+1+2)² / (1+1+4) = 16/6
    assert_allclose(kish_ess(np.log([1, 1, 2])), [1, 2, 16 / 6])


def test_nested_monitor():
    from numpy.testing import assert_allclose

    from .curve import Curve
    from .fit_thread import EVT_FIT_PROGRESS, NestedMonitor
    from .fitproblem import FitProblem
    from .fitters import DEFit, FitDriver, NestedSamplerFit

    x = np.array([1, 2, 3, 4, 5, 6.0])
    y = np.array([2.1, 4.0, 6.3, 8.03, 9.6, 11.9])
    model = Curve(lambda x, m, b: m * x + b, x, y, 0.05 * y, m=2, b=2)
    model.m.range(0, 4)
    model.b.range(-5, 5)

    def fit(fitclass, **options):
        """Fit with an update sent at every iteration, returning the driver and the stats sent."""
        sent = []

        def receive(event):
            if event["message"] == NestedMonitor.message:
                sent.append(event["stats"])

        EVT_FIT_PROGRESS.connect(receive)
        try:
            driver = FitDriver(fitclass, problem=FitProblem(model), monitors=[NestedMonitor(rate=1e-9)], **options)
            driver.fit()
        finally:
            EVT_FIT_PROGRESS.disconnect(receive)
        return driver, sent

    # A loose initial run followed by batches until the ESS target is met.
    driver, sent = fit(NestedSamplerFit, pop=-50, dlogz=1.0, pfrac=1.0, n_effective=500)
    phases = [stats.phase for stats in sent]
    assert phases[0] == "initial" and "batch" in phases and phases[-1] == "done"
    assert all(isinstance(stats, NestedStats) for stats in sent)
    final = sent[-1]
    assert len(final.logz) == len(final.logzerr) == len(final.logvol) == len(final.ess)
    assert_allclose(final.ess[-1], driver.fitter.nested_sampler.n_effective)
    assert final.stop is not None and final.stop <= 1

    # Other fitters send nothing. DE also replaces the history traces with its own.
    assert fit(DEFit, steps=5)[1] == []
