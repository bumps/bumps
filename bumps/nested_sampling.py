"""
Samples from a posterior distribution produced by a nested sampler.
"""

__all__ = ["NestedDraw"]

from typing import TYPE_CHECKING, Optional

import numpy as np

from .bayes import BayesDraw

if TYPE_CHECKING:
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

    @property
    def nfev(self) -> Optional[int]:
        """Number of likelihood evaluations, or None if not from dynesty."""
        return None if self.results is None else int(np.sum(self.results.ncall))

    def save(self, filename: str):
        """
        Save the sample to *filename* + "-nested.mc.gz".

        The file holds the unequally weighted nested sample, one row per
        sampled point: the named columns of the header followed by the point
        itself. This is the sample the evidence is computed from; the
        equally weighted posterior used for statistics and plots is a
        resampling of it.

        *logz* +/- *logzerr* is the running log of the Bayesian evidence and
        *information* the running information gained from prior to posterior
        (the KL divergence) in nats, so the estimates for the complete run
        are in the last row.

        Log Z is unnormalised: bumps computes the negative log likelihood as
        1/2 sum(residuals**2), dropping the gaussian normalisation constant
        -sum(log(sqrt(2 pi) dy)). Bayes factors are therefore only valid
        between models fitted to the same data with the same uncertainties.
        """
        from .dream.state import CREATE, EXT

        results = self.results
        if results is None:
            points, logl, weights = self.sample_points(portion=1.0)
            logwt = np.zeros(len(logl)) if weights is None else np.log(weights)
            columns = {"logl": logl, "logwt": logwt}
        else:
            points = results.samples
            columns = {
                "logl": results.logl,
                "logwt": results.logwt,
                "logvol": results.logvol,
                "logz": results.logz,
                "logzerr": results.logzerr,
                "information": results.information,
            }
            # number of live points per sample; dynamic runs only
            if "samples_n" in results:
                columns["nlive"] = results.samples_n
        table = np.hstack([np.asarray(v, "d").reshape(-1, 1) for v in columns.values()] + [points])
        with CREATE(filename + "-nested" + EXT, "wb") as fid:
            fid.write(f"# {' '.join(columns)} {points.shape[1]}*point\n".encode())
            np.savetxt(fid, table)
