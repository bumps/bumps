r"""
Importance sampling from the covariance matrix at the best fit.

Given the best fit point $\mu$ and its covariance matrix $\Sigma$, draw points
$x_k$ from a proposal distribution $Q$ with mean $\mu$ and covariance $\Sigma$,
and give each point the importance weight $w_k \propto P(x_k)/Q(x_k)$, where
$P(x) \propto e^{-\text{nllf}(x)}$ is the posterior probability. The weighted
sample represents the posterior distribution, so it can be analyzed with the
same statistics and plots as an MCMC sample, but unlike the square root of the
diagonal of the covariance matrix, it reflects the actual shape of the
likelihood. The weights are normalized by their sum, so constant offsets in
nllf don't matter.

The proposal is a multivariate t distribution with $\nu$ degrees of freedom and
scale matrix $\Sigma' = \Sigma (\nu - 2)/\nu$. A point is drawn as
$x = \mu + z \sqrt{\nu/y}$ with $z \sim N(0, \Sigma')$ and
$y \sim \chi^2_\nu$, which gives the proposal the covariance $\Sigma$. The t
distribution is a mixture of normal distributions with different widths, so
it keeps the weights bounded when the posterior is wider or narrower than the
covariance matrix suggests, or has heavier tails than a normal distribution.
The price is a lower effective sample size when the covariance matrix is
already exact, more so with many parameters. Use $\nu = 0$ for a normal
proposal.

Points outside the parameter bounds are redrawn rather than evaluated, since
they have zero posterior probability.

The effective sample size, $\text{ESS} = (\sum w)^2 / \sum w^2$, measures how
well the proposal matches the posterior. It is close to the number of samples
when the covariance matrix describes the posterior well. The uncertainty in
the derived statistics scales as $1/\sqrt{\text{ESS}}$.
"""

__all__ = ["Proposal", "ImportanceSample", "draw_within_bounds", "batch_size", "h5dump", "h5load"]

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp

logger = logging.getLogger(__name__)

LOW_ESS_FRACTION = 0.1
"""Warn when the effective sample size is below this fraction of the number of samples."""

NOT_AT_MINIMUM = 1.0
"""Warn that the fit was not at the minimum when a sample improves nllf by more than this."""

_MAX_CHUNK = 100_000
"""Maximum number of candidate points drawn from the proposal at once."""


class Proposal:
    """
    Multivariate t distribution with mean *mu*, covariance *cov* and *nu* degrees of freedom.

    The scale matrix is *cov* (nu - 2)/nu so that the covariance of the
    distribution is *cov*, which requires nu > 2. Use nu=0 for a multivariate
    normal distribution.
    """

    def __init__(self, mu: NDArray, cov: NDArray, nu: float):
        self.mu = np.asarray(mu, "d")
        self.cov = np.asarray(cov, "d")
        self.nu = float(nu)
        if self.nu != 0 and not self.nu > 2:
            raise ValueError(f"nu={nu} must be greater than 2, or 0 for a normal distribution")
        scale = self.cov if self.nu == 0 else self.cov * ((self.nu - 2) / self.nu)
        try:
            self._chol = np.linalg.cholesky(scale)
        except np.linalg.LinAlgError:
            raise ValueError(
                "The covariance matrix is not positive definite, so it cannot define the proposal"
                " distribution. Fit to the minimum first, and check for parameters that don't"
                " affect the fit."
            ) from None
        d = len(self.mu)
        log_det = 2 * np.sum(np.log(np.diag(self._chol)))
        if self.nu == 0:
            self._log_norm = -0.5 * (d * np.log(2 * np.pi) + log_det)
        else:
            nu = self.nu
            self._log_norm = gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * (d * np.log(nu * np.pi) + log_det)

    def draw(self, n: int) -> Tuple[NDArray, NDArray]:
        """
        Draw *n* points from the distribution.

        Returns the points and the log probability density at each point.
        """
        d = len(self.mu)
        z = np.random.standard_normal((n, d))
        # Squared Mahalanobis distance from the mean, relative to the scale matrix.
        r2 = np.sum(z**2, axis=1)
        if self.nu == 0:
            logq = self._log_norm - 0.5 * r2
        else:
            # Scale each normal draw by sqrt(nu/y) with y ~ chisq(nu).
            s2 = self.nu / np.random.chisquare(self.nu, size=n)
            z *= np.sqrt(s2)[:, None]
            r2 *= s2
            logq = self._log_norm - 0.5 * (self.nu + d) * np.log1p(r2 / self.nu)
        return self.mu + z @ self._chol.T, logq


def draw_within_bounds(
    proposal: Proposal, n: int, bounds: NDArray, max_draws: int
) -> Tuple[NDArray, NDArray, int, int]:
    """
    Draw *n* points from *proposal* that lie within *bounds* = [lower, upper].

    Rejecting points outside the bounds truncates the proposal to the region
    where the posterior can be nonzero. Truncation only changes the
    normalization of the proposal, which cancels when the weights are
    normalized.

    Returns *(points, logq, drawn, accepted)*, where *drawn* is the number of
    candidates drawn and *accepted* is how many of them were within the bounds.
    At most *max_draws* candidates are drawn, so fewer than *n* points are
    returned if the proposal rarely falls within the bounds.
    """
    lower, upper = bounds
    points, logq = [], []
    drawn = accepted = 0
    while accepted < n and drawn < max_draws:
        # Draw enough candidates to fill the request at the acceptance rate seen so far.
        rate = max(accepted, 1) / max(drawn, 1)
        size = min(int((n - accepted) / rate) + 1, max_draws - drawn, _MAX_CHUNK)
        x, lq = proposal.draw(size)
        keep = np.all((x >= lower) & (x <= upper), axis=1)
        points.append(x[keep])
        logq.append(lq[keep])
        drawn += size
        accepted += int(np.sum(keep))
    if not points:
        return np.empty((0, len(proposal.mu))), np.empty(0), drawn, accepted
    return np.vstack(points)[:n], np.hstack(logq)[:n], drawn, accepted


def batch_size(nvar: int, samples: int) -> int:
    """
    Number of points to evaluate in each step when drawing *samples* points.

    Use at least 10 points per parameter so that parallel evaluation stays
    efficient, and at most about 100 steps.
    """
    return max(10 * nvar, -(-samples // 100), 1)


class ImportanceSample:
    r"""
    Weighted sample of the posterior distribution from importance sampling.

    *proposal* is the distribution that the points are drawn from, *labels* are
    the parameter names and *title* is the name of the fit problem.

    Points are added with :meth:`append`. For each point, *points* holds the
    parameter values, *logp* holds $-\text{nllf}$, and *logq* holds the log
    density of the proposal. :attr:`weights` gives the importance weights and
    :attr:`ess` gives the effective sample size.

    The sample can be used in place of the MCMC state from DREAM: :meth:`draw`
    returns a :class:`bumps.dream.state.Draw` that carries the weights, for use
    with :func:`bumps.dream.stats.var_stats` and the uncertainty plots.
    """

    def __init__(self, proposal: Proposal, labels: List[str], title: Optional[str] = None):
        self.proposal = proposal
        self.labels = list(labels)
        self.title = title
        self.points = np.empty((0, len(self.labels)))
        self.logp = np.empty(0)
        self.logq = np.empty(0)
        # Candidate points drawn from the proposal, and how many were within the bounds.
        self.drawn = 0
        self.accepted = 0
        self.best_x = self.proposal.mu.copy()
        self.best_logp = -np.inf
        # -nllf at the center of the proposal, if known.
        self.start_logp = -np.inf
        self._weights = None
        # Variable selection and derived variables, used by Draw as in MCMCDraw.
        self._good_chains = slice(None)
        self._shown = None
        self._integer_vars = None
        self._derived_fn = None
        self._derived_labels = None

    def append(self, points: NDArray, logp: NDArray, logq: NDArray, drawn: int, accepted: int):
        r"""
        Add *points* drawn from the proposal.

        *logp* is $-\text{nllf}$ at each point and *logq* is the log density of the
        proposal. *drawn* is the number of candidates drawn from the proposal to
        get these points, and *accepted* is how many of them were within the
        bounds.
        """
        logp = np.array(logp, "d")
        # Treat failed model evaluations as having zero probability.
        logp[np.isnan(logp)] = -np.inf
        self.points = np.vstack((self.points, points))
        self.logp = np.hstack((self.logp, logp))
        self.logq = np.hstack((self.logq, logq))
        self.drawn += drawn
        self.accepted += accepted
        self._weights = None
        if len(logp):
            k = np.argmax(logp)
            self.update_best(points[k], logp[k])

    def update_best(self, x: NDArray, logp: float):
        """Record *x* as the best point if its *logp* is higher than the best seen so far."""
        if logp > self.best_logp:
            self.best_x, self.best_logp = np.array(x, "d"), float(logp)

    def best(self) -> Tuple[NDArray, float]:
        """Return the best point seen and its log likelihood."""
        return self.best_x, self.best_logp

    @property
    def Nsamples(self) -> int:
        """Number of points in the sample."""
        return len(self.logp)

    @property
    def weights(self) -> NDArray:
        """
        Importance weights $P/Q$ for each point, scaled to have mean 1.

        Points with zero posterior probability, such as those that fail the
        constraints, have zero weight.
        """
        if self._weights is None:
            log_w = self.logp - self.logq
            top = np.max(log_w, initial=-np.inf)
            if not np.isfinite(top):
                raise ValueError("No samples have a finite likelihood")
            w = np.exp(log_w - top)
            self._weights = w * (len(w) / np.sum(w))
        return self._weights

    @property
    def ess(self) -> float:
        r"""Effective sample size $(\sum w)^2/\sum w^2$ of the importance weights."""
        w = self.weights
        return float(np.sum(w) ** 2 / np.sum(w**2))

    def summary(self) -> str:
        """Describe the number of samples and the effective sample size, with a warning if it is low."""
        n, ess = self.Nsamples, self.ess
        lines = [f"Importance sampling: {n} samples, effective sample size {ess:.0f} ({100 * ess / n:.1f}%)"]
        if self.accepted < self.drawn:
            rejected = 1 - self.accepted / self.drawn
            lines.append(f"{100 * rejected:.1f}% of the points drawn were outside the parameter bounds and redrawn")
        improvement = self.best_logp - self.start_logp
        if np.isfinite(improvement) and improvement > NOT_AT_MINIMUM:
            lines.append(
                f"Warning: a sample improves nllf by {improvement:.3g} over the starting point, so the fit"
                " was not at the minimum. Run an optimizer such as lm before importance sampling."
            )
        if ess < LOW_ESS_FRACTION * n:
            lines.append(
                "Warning: the effective sample size is low, so the covariance matrix does not describe"
                " the posterior well and these uncertainties are unreliable. Use dream instead."
            )
        return "\n".join(lines)

    # The following methods let Draw treat the sample like MCMC chains with one chain.
    def chains(self) -> Tuple[None, NDArray, NDArray]:
        """Return the points as a single chain, in the form of :meth:`bumps.dream.state.MCMCDraw.chains`."""
        return None, self.points[:, None, :], self.logp[:, None]

    def chain_weights(self) -> NDArray:
        """Return the weight of each point in the shape of the logp array from :meth:`chains`."""
        return self.weights[:, None]

    def draw(
        self,
        portion: Optional[float] = None,
        vars: Optional[List[int]] = None,
        exclude: Optional[List[int]] = None,
        selection=None,
        thin: int = 1,
        outliers: bool = False,
        derived: Optional[Callable] = None,
    ):
        """
        Return the weighted sample as a :class:`bumps.dream.state.Draw`.

        The arguments are those of :meth:`bumps.dream.state.MCMCDraw.draw`. The
        points are independent, so *portion* and *thin* just use fewer of them,
        and *outliers* has no effect.
        """
        from .dream.state import Draw

        return Draw(
            state=self,
            portion=1.0 if portion is None else portion,
            vars=vars if vars is not None else self._shown,
            exclude=exclude,
            selection=selection,
            thin=thin,
            outliers=outliers,
            derived=derived,
        )

    def set_visible_vars(self, labels: List[str]):
        """Only show the parameters in *labels* in the statistics and plots."""
        self._shown = [self.labels.index(v) for v in labels]

    def set_integer_vars(self, labels: List[str]):
        """Treat the parameters in *labels* as integers when computing statistics."""
        self._integer_vars = np.array([var in labels for var in self.labels])

    def set_derived_vars(self, fn: Callable[[NDArray], NDArray], labels: List[str]):
        """Add variables computed by *fn* to each draw, as in :meth:`bumps.dream.state.MCMCDraw.set_derived_vars`."""
        self._derived_fn = fn
        self._derived_labels = labels

    def entropy(self, method: Optional[str] = None, n_bootstrap: int = 20) -> Tuple[float, float]:
        r"""
        Return the entropy of the posterior distribution in bits, and its uncertainty.

        Importance sampling estimates the normalization constant
        $Z = \int e^{-\text{nllf}(x)} dx$ directly, which gives the entropy
        $H = \log Z + E_P[\text{nllf}]$. The uncertainty comes from a bootstrap
        over the samples. The *method* argument is accepted for compatibility
        with :meth:`bumps.dream.state.MCMCDraw.entropy`, but other methods are not
        available for importance sampling.
        """
        if method not in (None, "is"):
            logger.warning(f"Using the importance sampling estimate of entropy instead of method={method!r}")
        valid = np.isfinite(self.logp)
        log_u = np.where(valid, self.logp - self.logq, -np.inf)
        nllf = np.where(valid, -self.logp, 0.0)
        # The proposal is truncated to the bounds, so its density is q/(accepted/drawn).
        log_accept = np.log(self.accepted / self.drawn)

        def estimate(index):
            lu, f = log_u[index], nllf[index]
            log_Z = logsumexp(lu) - np.log(len(lu)) + log_accept
            w = np.exp(lu - np.max(lu))
            return (log_Z + np.sum(w * f) / np.sum(w)) / np.log(2)

        n = self.Nsamples
        S = estimate(slice(None))
        bootstrap = [estimate(np.random.randint(n, size=n)) for _ in range(n_bootstrap)]
        return S, float(np.std(bootstrap, ddof=1))

    def show(self, portion: Optional[float] = None, figfile: Optional[str] = None):
        """Print the parameter statistics and plot the uncertainty, saving the plots if *figfile* is given."""
        from .dream.views import plot_all

        plot_all(self.draw(portion=portion), figfile=figfile)
        print(self.summary())

    def save(self, filename: str):
        """Write the weight, logp and parameter values of each point to *filename*-importance.dat.gz."""
        header = "\t".join(["weight", "logp", *self.labels])
        data = np.column_stack((self.weights, self.logp, self.points))
        np.savetxt(f"{filename}-importance.dat.gz", data, delimiter="\t", header=header)


def h5dump(group, state: ImportanceSample):
    """Store *state* in the HDF5 *group*."""
    from .state import write_json, write_ndarray, write_version

    write_version(group, (1, 0))
    write_json(group, "labels", state.labels)
    write_json(group, "title", state.title)
    write_json(group, "nu", state.proposal.nu)
    write_json(group, "drawn", state.drawn)
    write_json(group, "accepted", state.accepted)
    write_json(group, "best_logp", state.best_logp)
    write_json(group, "start_logp", state.start_logp)
    write_ndarray(group, "best_x", state.best_x)
    write_ndarray(group, "mu", state.proposal.mu)
    write_ndarray(group, "cov", state.proposal.cov)
    write_ndarray(group, "points", state.points)
    write_ndarray(group, "logp", state.logp)
    write_ndarray(group, "logq", state.logq)


def h5load(group) -> ImportanceSample:
    """Load the :class:`ImportanceSample` stored in the HDF5 *group* by :func:`h5dump`."""
    from .state import read_json, read_ndarray

    labels = read_json(group, "labels")
    proposal = Proposal(read_ndarray(group, "mu"), read_ndarray(group, "cov"), read_json(group, "nu"))
    state = ImportanceSample(proposal, labels, title=read_json(group, "title"))
    points = read_ndarray(group, "points")
    if points is not None:
        state.append(points, read_ndarray(group, "logp"), read_ndarray(group, "logq"), drawn=0, accepted=0)
    state.drawn = read_json(group, "drawn")
    state.accepted = read_json(group, "accepted")
    state.best_x = read_ndarray(group, "best_x")
    state.best_logp = read_json(group, "best_logp")
    state.start_logp = read_json(group, "start_logp") if "start_logp" in group else -np.inf
    return state


def test_proposal():
    from scipy.stats import multivariate_normal, multivariate_t

    np.random.seed(1)
    mu = np.array([1.0, -2.0, 0.5])
    cov = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, -0.4], [0.0, -0.4, 0.5]])
    for nu in (0.0, 5.0, 10.0):
        proposal = Proposal(mu, cov, nu)
        x, logq = proposal.draw(200_000)
        if nu == 0:
            expected = multivariate_normal(mu, cov).logpdf(x)
        else:
            expected = multivariate_t(mu, cov * (nu - 2) / nu, df=nu).logpdf(x)
        assert np.allclose(logq, expected)
        # Scaling the shape matrix by (nu - 2)/nu gives the requested covariance.
        assert np.allclose(np.cov(x.T), cov, rtol=0.03, atol=0.01)

    for nu in (-1.0, 1.0, 2.0):
        try:
            Proposal(mu, cov, nu)
        except ValueError:
            pass
        else:
            raise AssertionError(f"nu={nu} should be rejected")


def test_gaussian_posterior():
    from .dream.entropy import cov_entropy
    from .dream.stats import var_stats

    np.random.seed(2)
    # The posterior is N(m, C), but the proposal is off-center, 20% too narrow in a
    # and 20% too wide in b.
    m = np.array([1.0, 2.0])
    C = np.array([[0.04, 0.018], [0.018, 0.0225]])
    D = np.diag([0.8, 1.2])
    Cinv = np.linalg.inv(C)
    state = ImportanceSample(Proposal(m + [0.05, -0.03], D @ C @ D, nu=10), ["a", "b"])
    x, logq = state.proposal.draw(100_000)
    dx = x - m
    nllf = 0.5 * np.einsum("ij,jk,ik->i", dx, Cinv, dx)
    state.append(x, -nllf, logq, drawn=len(x), accepted=len(x))
    assert 0.3 * len(x) < state.ess < len(x)

    draw = state.draw()
    assert np.allclose(np.average(draw.points, axis=0, weights=draw.weights), m, atol=0.005)
    assert np.allclose(np.cov(draw.points.T, aweights=draw.weights), C, rtol=0.05)
    for v, mk, sk in zip(var_stats(draw), m, np.sqrt(np.diag(C))):
        assert abs(v.mean - mk) < 0.03 * sk
        assert abs(v.std - sk) < 0.03 * sk
        assert np.allclose(v.p68, [mk - sk, mk + sk], atol=0.03 * sk)

    # Selections keep the weights aligned with the points.
    selected = state.draw(selection={"a": (0.9, 1.1)})
    assert len(selected.weights) == len(selected.points) < len(draw.points)

    S, dS = state.entropy()
    assert abs(S - cov_entropy(C)) < 0.03 and 0 < dS < 0.03


def test_bounds():
    np.random.seed(3)
    # A half-normal posterior on x >= 0 sampled from a normal proposal truncated to the bounds.
    proposal = Proposal([0.0], [[1.0]], nu=0)
    x, logq, drawn, accepted = draw_within_bounds(proposal, 50_000, np.array([[0.0], [np.inf]]), max_draws=10**6)
    assert len(x) == 50_000 and np.all(x >= 0)
    assert abs(accepted / drawn - 0.5) < 0.01
    state = ImportanceSample(proposal, ["x"])
    state.append(x, -0.5 * x[:, 0] ** 2, logq, drawn=drawn, accepted=accepted)
    # The posterior matches the truncated proposal, so the weights are equal.
    assert np.allclose(state.weights, 1.0)
    assert "outside the parameter bounds" in state.summary()
    # The entropy accounts for the truncation: a half normal has one bit less than a normal.
    S, _ = state.entropy()
    assert abs(S - (0.5 * np.log2(2 * np.pi * np.e) - 1)) < 0.02

    # Give up when the proposal never falls within the bounds.
    x, _, drawn, accepted = draw_within_bounds(proposal, 10, np.array([[50.0], [51.0]]), max_draws=1000)
    assert len(x) == 0 and drawn == 1000 and accepted == 0


def test_low_ess_warning():
    np.random.seed(4)
    # The proposal is narrow and centered two standard deviations from the posterior mean,
    # so the few points nearest the mean carry most of the weight.
    state = ImportanceSample(Proposal([2.0], [[0.25]], nu=0), ["x"])
    x, logq = state.proposal.draw(1000)
    state.append(x, -0.5 * x[:, 0] ** 2, logq, drawn=len(x), accepted=len(x))
    assert state.ess < LOW_ESS_FRACTION * len(x)
    assert "Warning" in state.summary()


def test_h5_round_trip():
    import h5py

    np.random.seed(5)
    proposal = Proposal([1.0, 2.0], [[1.0, 0.2], [0.2, 0.5]], nu=10)
    state = ImportanceSample(proposal, ["a", "b"], title="test")
    x, logq = proposal.draw(1000)
    logp = -0.5 * np.sum((x - [1.0, 2.0]) ** 2, axis=1)
    logp[:10] = -np.inf
    state.append(x, logp, logq, drawn=1200, accepted=1000)
    state.start_logp = -1.5
    with h5py.File("round_trip.h5", "w", driver="core", backing_store=False) as fid:
        h5dump(fid.create_group("fit_state"), state)
        loaded = h5load(fid["fit_state"])
    assert loaded.labels == state.labels and loaded.title == state.title
    assert (loaded.drawn, loaded.accepted) == (1200, 1000)
    assert np.array_equal(loaded.points, state.points)
    assert np.array_equal(loaded.weights, state.weights)
    assert np.array_equal(loaded.proposal.cov, proposal.cov) and loaded.proposal.nu == proposal.nu
    assert np.array_equal(loaded.best_x, state.best_x) and loaded.best_logp == state.best_logp
    assert loaded.start_logp == state.start_logp


def test_fit():
    from .curve import Curve
    from .errplot import error_points_from_state
    from .fitproblem import FitProblem
    from .fitters import fit

    np.random.seed(6)
    x = np.linspace(0, 1, 11)
    dy = 0.1 * np.ones_like(x)
    y = 2 * x + 1 + dy * np.random.standard_normal(len(x))

    def line(x, m, b):
        return m * x + b

    M = Curve(line, x, y, dy, m=2, b=1)
    M.m.range(0, 4)
    M.b.range(-5, 5)
    problem = FitProblem(M)
    best = fit(problem, method="lm")
    result = fit(problem, method="importance", samples=2000)
    state = result.state
    assert isinstance(state, ImportanceSample) and state.Nsamples == 2000
    assert np.allclose(result.x, best.x, rtol=1e-6)
    # The model is linear, so the proposal matches the posterior up to its tails.
    assert state.ess > 0.8 * state.Nsamples
    assert np.allclose(result.dx, best.dx, rtol=0.1)

    # Resuming adds samples from the same proposal.
    resumed = fit(problem, method="importance", samples=1000, resume=result)
    assert resumed.state.Nsamples == 3000

    # Model uncertainty plots draw points from the weighted sample.
    points = error_points_from_state(resumed.state, nshown=20)
    assert points.shape == (20, 2)

    # Starting away from the minimum is reported.
    assert "not at the minimum" not in state.summary()
    problem.setp(best.x + 5 * best.dx)
    offset = fit(problem, method="importance", samples=2000)
    assert "not at the minimum" in offset.state.summary()
