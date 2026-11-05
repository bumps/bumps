"""
Samples from a posterior distribution produced by a nested sampler.
"""

__all__ = ["NestedDraw"]

from .bayes import BayesDraw


class NestedDraw(BayesDraw):
    """
    A sample from a posterior distribution produced by a nested sampler.

    The points are equally weighted and unordered, so the statistics and
    plots for an ordinary posterior sample apply unchanged. The chain
    history plots do not, and are skipped for anything which is not an
    :class:`~bumps.dream.state.MCMCDraw`.
    """
