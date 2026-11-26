"""
Samples from a posterior distribution produced by a nested sampler.
"""

__all__ = ["NestedDraw", "capture_sampler_state", "restore_sampler"]

import json
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .bayes import BayesDraw

if TYPE_CHECKING:
    from dynesty import DynamicNestedSampler
    from dynesty.utils import Results, RunRecord
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

    *sampler_state* is the rest of the sampler state needed to continue the
    run from *results*; see :func:`capture_sampler_state`.
    """

    results: Optional["Results"] = None
    sampler_state: dict[str, Any] | None = None

    def __init__(self, *args, results: Optional["Results"] = None, sampler_state: dict[str, Any] | None = None, **kw):
        super().__init__(*args, **kw)
        self.results = results.copy() if results is not None else None
        self.sampler_state = sampler_state

    @property
    def nfev(self) -> Optional[int]:
        """Number of likelihood evaluations, or None if not from dynesty."""
        return None if self.results is None else int(np.sum(self.results.ncall))

    @property
    def nlive(self) -> int:
        """Number of live points in the initial run."""
        if "live_u" in self.sampler_state:
            return len(self.sampler_state["live_u"])
        return int(self.results.batch_nlive[0])

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
        Store the sample in the given HDF5 group.
        """
        from .dream.state import LABEL_DTYPE

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
        _write_fields(group, fields)
        if self.results is not None:
            # dynesty results
            _write_fields(group.create_group("results"), self.results)
        if self.sampler_state is not None:
            # sampler state (for resuming)
            _write_fields(group.create_group("sampler"), self.sampler_state)

    @classmethod
    def h5load(cls, group: "Group") -> "NestedDraw":
        """
        Load a sample from the HDF5 *group* written by :meth:`h5dump`.
        """
        fields = _read_fields(group)
        results = None
        if "results" in group:
            from dynesty.utils import Results

            results = Results(_read_fields(group["results"]))
        sampler_state = _read_fields(group["sampler"]) if "sampler" in group else None

        title = fields.get("title")
        best_x = fields.get("best_x")
        return cls(
            points=fields["points"],
            logp=fields["logp"],
            weights=fields.get("weights"),
            labels=[label.decode() for label in fields["labels"]],
            title=title,
            portion=fields.get("portion", 1.0),
            best=None if best_x is None else (best_x, fields["best_logp"]),
            results=results,
            sampler_state=sampler_state,
        )


def _write_fields(group: "Group", fields: dict[str, Any]):
    """
    Write *fields* to the HDF5 *group*, skipping None and object arrays.
    """
    from .dream.state import _h5_write_field

    for key, value in fields.items():
        value = np.asarray(value) if isinstance(value, (list, tuple)) else value
        if value is not None and not (isinstance(value, np.ndarray) and value.dtype.kind == "O"):
            _h5_write_field(group, key, value)


def _read_fields(group: "Group") -> dict[str, Any]:
    """
    Read the datasets of the HDF5 *group* written by :func:`_write_fields`.
    """
    from h5py import Dataset

    from .dream.state import _h5_read_field

    fields = {key: _h5_read_field(group, key) for key, item in group.items() if isinstance(item, Dataset)}
    return {key: value.decode() if isinstance(value, bytes) else value for key, value in fields.items()}


#: Columns of the dynesty run record, with the results entries they come from.
_RUN_COLUMNS = {
    "id": "samples_id",
    "u": "samples_u",
    "v": "samples",
    "logl": "logl",
    "logvol": "logvol",
    "logwt": "logwt",
    "logz": "logz",
    "h": "information",
    "nc": "ncall",
    "it": "samples_it",
    "n": "samples_n",
    "scale": "scale",
    "batch": "samples_batch",
    "batch_nlive": "batch_nlive",
    "batch_logl_bounds": "batch_logl_bounds",
}


def capture_sampler_state(sampler: "DynamicNestedSampler") -> dict[str, Any]:
    """
    Gather what *sampler* needs to resume a run.
    """
    from dynesty.dynamicsampler import DynamicSamplerStatesEnum

    inner = sampler.sampler
    state = {
        "rng": json.dumps(sampler.rstate.bit_generator.state),
        "logl_first_update": inner.logl_first_update,
    }
    if sampler.internal_state == DynamicSamplerStatesEnum.INBASE:
        state.update(
            live_u=np.array(inner.live_u),
            live_v=np.array(inner.live_v),
            live_logl=np.array(inner.live_logl),
            live_it=np.array(inner.live_it),
            plateau_counter=inner.plateau_counter if inner.plateau_mode else 0,
            plateau_logdvol=inner.plateau_logdvol,
        )
    return state


def restore_sampler(sampler: "DynamicNestedSampler", draw: NestedDraw):
    """
    Apply the current state of the *draw* object to the sampler.
    """
    from dynesty.dynamicsampler import DynamicSamplerStatesEnum
    from dynesty.sampler import Sampler

    results, state, nlive = draw.results, draw.sampler_state, draw.nlive
    sampler.rstate.bit_generator.state = json.loads(state["rng"])

    initial = "live_u" in state
    if initial:
        live = [np.array(state[key]) for key in ("live_u", "live_v", "live_logl")]
    else:
        # The batches draw their live points from the saved run, so the inner
        # sampler is only consulted for its settings and these are never used.
        live = [np.array(results[key][-nlive:]) for key in ("samples_u", "samples", "logl")]
    inner = Sampler(
        sampler.loglikelihood,
        sampler.prior_transform,
        sampler.ndim,
        live,
        sampler.sampling,
        sampler.bounding,
        ncdim=sampler.ncdim,
        rstate=sampler.rstate,
        pool=sampler.pool,
        use_pool=sampler.use_pool,
        queue_size=sampler.queue_size,
        bound_update_interval=max(round(sampler.bound_update_interval_ratio * nlive), 1),
        first_bound_update=sampler.first_bound_update,
        bound_bootstrap=sampler.bound_bootstrap,
        bound_enlarge=sampler.bound_enlarge,
        blob=sampler.blob,
        logvol_init=results.logvol_init,
    )
    # Every point's first likelihood call is counted with the point it
    # replaces, so the calls for the live points are not yet in the results.
    ncall = int(np.sum(results.ncall)) + (nlive if initial else 0)
    sampler.nlive0 = sampler.nlive_init = nlive
    sampler.live_init = live
    sampler.logvol_init = results.logvol_init
    sampler.saved_run = _run_record(results)
    sampler.it = inner.it = int(results.niter) + 1
    sampler.ncall = inner.ncall = ncall
    sampler.eff = inner.eff = 100.0 * sampler.it / ncall
    sampler.save_bounds = False
    sampler.sampler = inner

    if initial:
        inner.saved_run = _run_record(results)
        inner.live_it = np.array(state["live_it"])
        inner.plateau_counter = state["plateau_counter"]
        inner.plateau_mode = inner.plateau_counter > 0
        inner.plateau_logdvol = state.get("plateau_logdvol")
        sampler.base_run = _run_record(results)
        sampler.internal_state = DynamicSamplerStatesEnum.INBASE
        if state.get("logl_first_update") is not None:
            # The run had left the unit cube, so sample from a bound fitted to the live points.
            inner.update_bound_if_needed(results.logl[-1], force=True)
    else:
        sampler.batch = len(results.batch_nlive) - 1
        sampler.internal_state = DynamicSamplerStatesEnum.BASE_DONE
    inner.logl_first_update = state.get("logl_first_update")
    sampler.bound_list = list(inner.bound_list)


def _run_record(results: "Results") -> "RunRecord":
    """
    Return the dynesty run record of the run in *results*.
    """
    from dynesty.utils import RunRecord

    count = len(results.logl)
    record = RunRecord(dynamic=True)
    for key, name in _RUN_COLUMNS.items():
        # empty entries are read back from HDF5 as None
        record[key] = [] if results[name] is None else list(results[name])
    for key in ("boundidx", "bounditer"):
        record[key] = [0] * count
    for key in ("blob", "proposal_stats"):
        record[key] = [None] * count
    return record


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
        sampler_state={
            "rng": '{"bit_generator": "PCG64"}',
            "live_u": rng.uniform(size=(5, 2)),
            "live_it": np.arange(5),
            "plateau_counter": 0,
            # None entries are dropped on save
            "plateau_logdvol": None,
        },
    )
    # a state without results or sampler state
    bare = NestedDraw(points=samples, logp=logl, labels=["a", "b"])

    with h5py.File("in-memory.h5", "w", driver="core", backing_store=False) as fid:
        state.h5dump(fid.create_group("fit_state"))
        loaded = NestedDraw.h5load(fid["fit_state"])
        bare.h5dump(fid.create_group("bare"))
        loaded_bare = NestedDraw.h5load(fid["bare"])

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
    assert loaded.sampler_state.keys() == state.sampler_state.keys() - {"plateau_logdvol"}
    for key, value in loaded.sampler_state.items():
        assert_array_equal(value, state.sampler_state[key])
    assert loaded_bare.results is None and loaded_bare.sampler_state is None
