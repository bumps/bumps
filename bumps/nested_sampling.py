"""
Samples from a posterior distribution produced by a nested sampler.
"""

__all__ = ["NestedDraw"]

from typing import TYPE_CHECKING, Optional

import numpy as np

from .bayes import BayesDraw

if TYPE_CHECKING:
    from dynesty.utils import Results
    from h5py import Group


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

    def h5dump(self, group: "Group"):
        """
        Store the sample in the HDF5 *group*.

        The dynesty results are stored in a "results" subgroup. Only the numeric
        entries are kept: the object entries (*bound*, *blob* and
        *proposal_stats*) can't be stored in HDF5 and aren't needed to plot or
        export the sample.
        """
        from .dream.state import LABEL_DTYPE, _h5_write_field

        points, logp, weights = self.sample_points(portion=1.0)
        best_x, best_logp = self.best()
        fields = {
            "points": points,
            "logp": logp,
            "weights": weights,
            "labels": np.array(self.labels, dtype=LABEL_DTYPE),
            "title": self.title,
            "portion": self.portion,
            "best_x": best_x,
            "best_logp": best_logp,
        }
        for field, value in fields.items():
            if value is not None:
                _h5_write_field(group, field, value)

        if self.results is not None:
            results_group = group.create_group("results")
            for key, value in self.results.items():
                value = np.asarray(value) if isinstance(value, (list, tuple)) else value
                if isinstance(value, np.ndarray) and value.dtype.kind == "O":
                    continue
                if value is not None:
                    _h5_write_field(results_group, key, value)

    @classmethod
    def h5load(cls, group: "Group") -> "NestedDraw":
        """
        Load a sample from the HDF5 *group* written by :meth:`h5dump`.
        """
        from .dream.state import _h5_read_field

        fields = {field: _h5_read_field(group, field) for field in group if field != "results"}
        results = None
        if "results" in group:
            from dynesty.utils import Results

            results_group = group["results"]
            results = Results({key: _h5_read_field(results_group, key) for key in results_group})

        title = fields.get("title")
        best_x = fields.get("best_x")
        return cls(
            points=fields["points"],
            logp=fields["logp"],
            weights=fields.get("weights"),
            labels=[label.decode() for label in fields["labels"]],
            title=title.decode() if isinstance(title, bytes) else title,
            portion=fields.get("portion", 1.0),
            best=None if best_x is None else (best_x, fields["best_logp"]),
            results=results,
        )


def test_h5_round_trip():
    """
    Test saving/loading the state restores the same state.
    """
    import h5py
    from dynesty.utils import Results
    from numpy.testing import assert_array_equal

    rng = np.random.default_rng(0)
    n = 20
    samples = rng.normal(size=(n, 2))
    logl = np.sort(rng.normal(size=n))
    results = Results(
        dict(
            niter=n,
            ncall=np.ones(n, dtype=np.int64),
            samples=samples,
            samples_u=rng.uniform(size=(n, 2)),
            samples_id=np.arange(n),
            samples_n=np.full(n, 10),
            logl=logl,
            logz=np.linspace(-5, -1, n),
            logzerr=np.full(n, 0.1),
            information=np.linspace(0, 2, n),
            # object entries are dropped on save
            blob=np.array([None] * n, dtype=object),
        )
    )
    state = NestedDraw(
        points=samples[::-1],
        logp=logl[::-1],
        labels=["a", "b"],
        title="model",
        best=(samples[0], 1.5),
        results=results,
    )

    with h5py.File("in-memory.h5", "w", driver="core", backing_store=False) as fid:
        state.h5dump(fid.create_group("fit_state"))
        loaded = NestedDraw.h5load(fid["fit_state"])

    points, logp, weights = loaded.sample_points(portion=1.0)
    assert_array_equal(points, samples[::-1])
    assert_array_equal(logp, logl[::-1])
    assert weights is None
    assert loaded.labels == ["a", "b"]
    assert loaded.title == "model"
    assert_array_equal(loaded.best()[0], samples[0])
    assert loaded.best()[1] == 1.5
    assert loaded.nfev == n
    for key in ("samples", "logl", "logz", "logzerr", "information", "samples_n"):
        assert_array_equal(loaded.results[key], results[key])
    assert "blob" not in loaded.results.keys()
