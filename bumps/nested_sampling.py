"""
Samples from a posterior distribution produced by a nested sampler.
"""

__all__ = ["NestedDraw"]

from typing import TYPE_CHECKING, Optional

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
