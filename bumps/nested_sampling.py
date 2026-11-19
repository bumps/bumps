"""
Samples from a posterior distribution produced by a nested sampler.
"""

__all__ = ["NestedDraw", "h5dump", "h5load"]

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from .bayes import BayesDraw
from .dream.state import LABEL_DTYPE, _h5_read_field, _h5_write_field

if TYPE_CHECKING:
    from dynesty.utils import Results
    from h5py import Group
    from numpy.typing import NDArray


# The remaining allowed keys hold python objects rather than numbers: *bound*
# is the list of bounding objects, which is only needed for dynesty's own
# bound plots, and *blob* and *proposal_stats* are sampler bookkeeping.
#: Keys of :class:`dynesty.utils.Results` which are kept with the sample.
RUN_FIELDS = frozenset(
    (
        "batch_logl_bounds",
        "batch_nlive",
        "bound_iter",
        "eff",
        "information",
        "logl",
        "logvol",
        "logwt",
        "logz",
        "logzerr",
        "ncall",
        "niter",
        "nlive",
        "samples",
        "samples_batch",
        "samples_bound",
        "samples_id",
        "samples_it",
        "samples_n",
        "samples_u",
        "scale",
    )
)


def _final(value) -> Optional[float]:
    """
    Return the last element of a per-iteration estimate, or None if it is
    missing.
    """
    # Dynesty reports the evidence once per iteration; the value for the run
    # as a whole is the one from the final iteration.
    return None if value is None else float(np.asarray(value).reshape(-1)[-1])


class NestedDraw(BayesDraw):
    """
    A sample from a posterior distribution produced by a nested sampler.

    The points are equally weighted and unordered, so the statistics and
    plots for an ordinary posterior sample apply unchanged. The chain
    history plots do not, and are skipped for anything which is not an
    :class:`~bumps.dream.state.MCMCDraw`.

    *results* is the raw :class:`dynesty.utils.Results` from the sampler,
    which holds the unequally weighted points and the evidence estimate.
    A copy is taken.

    *resample_index* selects the equal weight points from *results*, so that
    a draw which is saved and reloaded contains the same sample.

    *complete* is False for a sample taken from a run which was still in
    progress. Its :attr:`logz` is a lower bound and its posterior is missing
    the high likelihood core that the rest of the run would have found.
    """

    #: Estimated log evidence of the run, or None if it was not recorded.
    logz: Optional[float] = None
    #: Uncertainty in :attr:`logz`.
    logzerr: Optional[float] = None
    #: Kullback-Leibler divergence; measures how difference the prior is from the posterior.
    information: Optional[float] = None
    #: Whether the run which produced the sample ran to completion.
    complete: bool = True
    resample_index: Optional["NDArray"] = None

    def __init__(
        self,
        *args,
        results: Optional["Results"] = None,
        resample_index: Optional["NDArray"] = None,
        complete: bool = True,
        **kw,
    ):
        super().__init__(*args, **kw)
        self.resample_index = resample_index
        self.complete = complete
        # Copy the arrays so that they are not modified by a sampler which
        # keeps running. Scalars are immutable, and must stay scalars so that
        # they are not written to hdf5 as zero dimensional arrays.
        run = (
            {}
            if results is None
            else {k: np.copy(v) if np.ndim(v) else v for k, v in results.asdict().items() if k in RUN_FIELDS}
        )
        self._set_run(run)

    def _set_run(self, run: Dict[str, Any]):
        """
        Attach the raw per-iteration arrays of the run and the evidence they
        imply.
        """
        # store as plain arrays so dynesty isn't required to read the session file 
        self._run = run
        self._results = None
        self.logz = _final(run.get("logz"))
        self.logzerr = _final(run.get("logzerr"))
        self.information = _final(run.get("information"))

    @property
    def results(self) -> Optional["Results"]:
        """
        The run as a :class:`dynesty.utils.Results`, or None if the run was
        not recorded or dynesty is not installed.
        """
        if self._results is None and self._run:
            # Only this view of the run needs dynesty; the sample itself can
            # be examined without it.
            try:
                from dynesty.utils import Results
            except ImportError:
                return None

            self._results = Results(self._run)
        return self._results


def h5dump(group: "Group", draw: NestedDraw):
    """
    Write the nested sampling draw into an open hdf5 group.
    """
    import dynesty

    fields = dict(draw._run)
    fields.update(
        labels=np.array(draw.labels, dtype=LABEL_DTYPE),
        title=draw.title,
        portion=draw.portion,
        best_x=draw._best_x,
        best_logp=draw._best_logp,
        complete=draw.complete,
        resample_index=draw.resample_index,
        # record version for troubleshooting, as dynesty save state can change
        # across versions
        dynesty_version=dynesty.__version__,
    )
    for field, value in fields.items():
        # Empty values are written as nothing and read back as None.
        if value is None or np.size(value) == 0:
            continue
        try:
            _h5_write_field(group, field, value)
        except Exception as exc:
            exc.args = (f"{exc.args[0]} while writing {field}",) + exc.args[1:]
            raise


def h5load(group: "Group") -> NestedDraw:
    """
    Read a nested sampling draw from an open hdf5 group.
    """
    # Fields added in later versions should be read with a default
    # like in bumps.dream.state
    fields = {name: _h5_read_field(group, name) for name in group}
    fields = {name: value for name, value in fields.items() if value is not None}
    run = {name: value for name, value in fields.items() if name in RUN_FIELDS}

    index = fields.get("resample_index")
    title = fields.get("title")
    best_x = fields.get("best_x")
    draw = NestedDraw(
        points=run["samples"][index],
        logp=run["logl"][index],
        labels=[label.decode() for label in fields["labels"]],
        title=None if title is None else title.decode(),
        portion=fields.get("portion", 1.0),
        best=None if best_x is None else (best_x, fields.get("best_logp", -np.inf)),
        resample_index=index,
        complete=bool(fields.get("complete", True)),
    )
    draw._set_run(run)
    return draw
