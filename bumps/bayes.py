"""
Samples from a Bayesian posterior distribution.

:class:`BayesDraw` holds the points, their log likelihoods and
the labels naming the parameters. :class:`Draw` is a view of that sample with
derived variables added and a subset of the points and parameters selected.
"""

__all__ = ["BayesDraw", "Draw"]

from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import argmax, asarray, inf
from numpy.typing import NDArray

SelectionType = Optional[Dict[Union[int, str], Tuple[float, float]]]

PatternType = Union[int, str]


def _match(pattern: PatternType, index: int, label: str):
    return index == pattern or (isinstance(pattern, str) and fnmatch(label, pattern))


class BayesDraw:
    """
    A sample from a Bayesian posterior distribution.

    *points* [Nsamples x Nvar] are the sampled points, *logp* [Nsamples] is
    the log likelihood of each point, and *weights* [Nsamples] is the weight
    of each point, or None if the points are equally weighted.

    *labels* names each parameter, defaulting to P0, P1, ...

    *title* is a title for the plots.

    *portion* is the fraction of the sample to use for statistics and plots,
    taken from the end of the sample.

    *best* is the best *(point, log likelihood)* seen, defaulting to the best
    point of the sample.
    """

    _derived_fn: Optional[Callable[[NDArray], NDArray]] = None
    _derived_labels: Optional[List[str]] = None
    _labels: Optional[List[str]] = None
    _integer_vars: Optional[NDArray] = None
    _shown: Optional[List[int]] = None
    title: Optional[str] = None

    def __init__(
        self,
        points: Optional[NDArray] = None,
        logp: Optional[NDArray] = None,
        weights: Optional[NDArray] = None,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        portion: float = 1.0,
        best: Optional[Tuple[NDArray, float]] = None,
    ):
        self._points = points
        self._logp = logp
        self._weights = weights
        self._labels = labels
        self.title = title
        self.portion = portion
        if best is None and logp is not None:
            k = argmax(logp)
            best = points[k], logp[k]
        self._best_x, self._best_logp = best if best is not None else (None, -inf)

    @property
    def Nvar(self):
        """Number of parameters in each point."""
        return self._points.shape[1]

    @property
    def Nsamples(self):
        """Number of points in the sample."""
        return len(self._points)

    def sample_points(self, portion: Optional[float] = None, thin: int = 1, outliers: bool = False):
        """
        Return *(points, logp, weights)* for the sample.

        *portion* is the fraction of the sample to return, taken from the end.
        *thin* returns every nth point. *outliers* returns the points from
        chains which have been marked as outliers as well.
        """
        index = self._trimmed(portion, thin, self.Nsamples)
        weights = None if self._weights is None else self._weights[index]
        return self._points[index], self._logp[index], weights

    def _trimmed(self, portion: Optional[float], thin: int, n: int):
        portion = self.portion if portion is None else portion
        return slice(int((1 - portion) * n), None, thin)

    def _default_thin(self, n_est: int):
        return max(self.Nsamples // n_est, 1)

    def best(self):
        """Return the best point seen and its log likelihood."""
        return self._best_x, self._best_logp

    def stderr(self):
        """
        Return half the 68% credible interval of each parameter, which is a
        more robust measure of spread than the standard deviation when the
        distribution is not normal.
        """
        from .dream.stats import var_stats

        return np.array([(v.p68[1] - v.p68[0]) / 2 for v in var_stats(self.draw())], "d")

    @property
    def labels(self):
        """
        Names of the sampled parameters, excluding the derived parameters.
        For those, query *state.draw().labels*.
        """
        return self._labels if self._labels is not None else ["P%d" % k for k in range(self.Nvar)]

    @labels.setter
    def labels(self, v: List[str]):
        self._labels = v

    def show_labels(self):
        """
        Print the parameters with the index to use for them in *draw()*.
        """
        print("\n".join(f"p[{k}] = {v}" for k, v in enumerate(self.labels)))

    # TODO: Move processing of visible/integer/derived out of state
    def set_visible_vars(self, labels: List[str]):
        """
        Restrict the default draw to the named parameters.
        """
        self._shown = [self.labels.index(v) for v in labels]

    def set_integer_vars(self, labels: List[str]):
        """
        Treat the named parameters as integers when computing statistics.
        """
        self._integer_vars = np.array([var in labels for var in self.labels])

    def set_derived_vars(self, fn: Callable[[NDArray], NDArray], labels: List[str]):
        """
        Add derived parameters to the draw.

        *fn* maps the sampled points p[:, k] to one or more derived values and
        *labels* names them. For example, to add the variable x+y::

            state.set_derived_vars(lambda p: p[0] + p[1], labels=["x+y"])
        """
        self._derived_fn = fn
        self._derived_labels = labels

    def show(self, portion: Optional[float] = None, figfile: Union[str, Path, None] = None):
        from .dream.views import plot_all

        plot_all(self, portion=portion, figfile=figfile)

    def sample(self, **kw):
        """
        Return the points and log likelihoods of a draw.

        **Deprecated** use :meth:`draw` instead.
        """
        drawn = self.draw(**kw)
        return drawn.points, drawn.logp

    def draw(
        self,
        portion: Optional[float] = None,
        vars: Optional[List[PatternType]] = None,
        exclude: Optional[List[PatternType]] = None,
        selection: SelectionType = None,
        thin: int = 1,
        outliers: bool = False,
        derived: Optional[Callable] = None,
    ):
        """
        Return a :class:`Draw` of the posterior distribution.

        To plot the distribution for parameter p1::

            draw = state.draw()
            hist(draw.points[:, 0])

        To plot the interdependence of p1 and p2::

            draw = state.draw()
            plot(draw.points[:, 0], draw.points[:, 1], '.')
        """
        return Draw(
            state=self,
            portion=portion,
            vars=vars if vars is not None else self._shown,
            exclude=exclude,
            selection=selection,
            thin=thin,
            outliers=outliers,
            derived=derived,
        )

    def entropy(
        self,
        vars: Optional[List[PatternType]] = None,
        portion: Optional[float] = None,
        selection: SelectionType = None,
        n_est: int = 10000,
        thin: Optional[int] = None,
        method: Optional[str] = None,
    ):
        r"""
        Return the entropy estimate and its uncertainty.

        *vars*, *portion*, *selection* and *thin* restrict the draw used for
        the estimate, as for :meth:`draw`. Thinning defaults to whatever gives
        about *n_est* points.

        *n_est* is the number of points to use in the estimate.

        *method* determines which entropy calculation to use:

        * gmm: fit sample to a gaussian mixture model (GMM) with $5 \sqrt{d}$
          components where $d$ is the number fitted parameters and estimate
          entropy by sampling from the GMM.

        * llf: estimates likelihood scale factor from ratio of density
          estimate to model likelihood, then computes Monte Carlo entropy
          from sample; this does not work for marginal likelihood estimates.
          DOI:10.1109/CCA.2010.5611198

        * mvn: fit sample to a multi-variate Gaussian and return the entropy
          of the best fit gaussian; uses bootstrap to estimate uncertainty.

        * wnn: estimate entropy from nearest-neighbor distances in sample.
          DOI:10.1214/18-AOS1688
        """
        from .dream import entropy

        if thin is None:
            thin = self._default_thin(n_est)
        drawn = self.draw(portion=portion, vars=vars, selection=selection, thin=thin)

        # TODO: don't print within a library function!
        print("Entropy from MVN: %s" % str(entropy.MVNEntropy(drawn.points)))

        if method is None:
            # TODO: change default to gmm
            method = "llf"
        if method == "llf":
            return entropy.entropy(drawn.points, drawn.logp, N_entropy=n_est)
        if method == "gmm":
            return entropy.gmm_entropy(drawn.points, n_est=n_est)
        if method == "wnn":
            return entropy.wnn_entropy(drawn.points, n_est=n_est)
        if method == "mvn":
            return entropy.mvn_entropy_bootstrap(drawn.points)
        raise ValueError("unknown method %r" % method)


class Draw:
    """
    A view of a :class:`BayesDraw` with restrictions applied.

    *state* is the sample to draw from.

    *portion* is the fraction of the sample to use, taken from the end, or
    None for the portion stored in *state*.

    *vars* are the parameters to keep, given as 0-origin indices or as glob
    patterns matching the labels, with `*` matching any sequence, `?` matching
    any character, `[abc]` matching one of `abc` and `[!abc]` not matching one
    of `abc`. *exclude* drops parameters by the same rules, except those which
    *vars* names explicitly.

    *selection* restricts the points to a region of the parameter space using
    {var: (low, high)}, where var is an index, a label pattern, or "logp" to
    restrict by log likelihood. A label uses the first restriction it matches.

    *thin* keeps every nth point and *outliers* keeps the points from chains
    which have been marked as outliers.

    *derived* computes extra parameters from each point, overriding the
    function given to :meth:`BayesDraw.set_derived_vars`.

    **Attributes**

    *points* [n x Nvar], *logp* [n] and *weights* [n] are the selected points,
    their log likelihoods and their weights, with weights None if the points
    are equally weighted. There is no inherent ordering to the points.

    *labels*, *integers* and *Nvar* describe the parameters remaining after
    adding the derived parameters and dropping the unwanted ones. *integers*
    flags the parameters whose value should be floored before use.
    """

    state: BayesDraw
    vars: Optional[List[int]]
    portion: float
    selection: SelectionType
    thin: int
    Nvar: int

    title: Optional[str]
    labels: List[str]
    integers: NDArray
    logp: NDArray
    points: NDArray
    weights: Optional[NDArray]

    def __init__(
        self,
        state: BayesDraw,
        portion: Optional[float] = None,
        vars: Optional[List[PatternType]] = None,
        exclude: Optional[List[PatternType]] = None,
        selection: SelectionType = None,
        thin: int = 1,
        outliers: bool = False,
        derived: Optional[Callable] = None,
    ):
        self.state = state
        self.portion = state.portion if portion is None else portion
        self.thin = thin
        self.selection = selection
        self.title = state.title
        self.labels = state.labels[:]
        self.integers = (
            state._integer_vars if state._integer_vars is not None else np.zeros(len(self.labels), dtype=bool)
        )

        points, logp, weights = state.sample_points(portion=self.portion, thin=thin, outliers=outliers)

        # Derived parameters are added to the draw rather than the state so
        # that resume still works.
        if derived is None:
            derived = state._derived_fn
        if derived is not None:
            new = derived(points.T)
            new_labels = None
            if isinstance(new, dict):
                new_labels, new = list(new.keys()), list(new.values())
            new = asarray(new).T.reshape(len(points), -1)
            if new_labels is None:
                new_labels = state._derived_labels or [f"F{k+1}" for k in range(new.shape[1])]
            points = np.hstack((points, new))
            self.labels.extend(new_labels)
            self.integers = np.append(self.integers, [False] * new.shape[1])

        # Selection is by index into the full set of parameters, so it must
        # happen before the unwanted parameters are dropped.
        mask = None
        for var, (low, high) in self._selected(selection).items():
            values = logp if var == "logp" else points[:, var]
            within = (values >= low) & (values <= high)
            mask = within if mask is None else mask & within

        # TODO: with pattern matching we've lost the ability to reorder parameters
        include = self._matching(vars)
        exclude = [k for k in self._matching(exclude) if k not in include]
        if exclude:
            include = [k for k in include or range(len(self.labels)) if k not in exclude]
        if include:
            points = points[:, include]
            self.labels = [self.labels[k] for k in include]
            self.integers = self.integers[include]
        self.vars = include if include else None

        if mask is not None:
            points, logp = points[mask], logp[mask]
            weights = None if weights is None else weights[mask]

        self.points, self.logp, self.weights = points, logp, weights
        self.Nvar = points.shape[1]
        self._argsort_indices = {}

    def get_argsort_indices(self, var: int):
        """
        Sort var by value. Unlike argsort this caches the results for reuse.
        """
        if var not in self._argsort_indices:
            self._argsort_indices[var] = np.argsort(self.points[:, var].flatten())
        return self._argsort_indices[var]

    def _selected(self, selection: SelectionType):
        """Limits from *selection*, keyed by parameter index or "logp"."""
        if selection is None:
            return {}
        limits = {"logp": selection["logp"]} if "logp" in selection else {}
        for index, label in enumerate(self.labels):
            for pattern, data in selection.items():
                if _match(pattern, index, label):
                    limits[index] = data
                    break
        return limits

    def _matching(self, patterns: Optional[List[PatternType]]):
        """Indices of the parameters matched by *patterns*."""
        if patterns is None:
            return []
        return [k for k, label in enumerate(self.labels) if any(_match(p, k, label) for p in patterns)]
