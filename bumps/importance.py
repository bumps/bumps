r"""
Parameter uncertainty from a likelihood-weighted sample of the covariance matrix.

Fits without an MCMC sample have only the covariance matrix $\Sigma$ at the best
fit $\mu$ to describe the parameter uncertainty, and the square root of its
diagonal assumes the posterior is gaussian. :func:`sample_from_cov` improves on
this by drawing points $x_k$ from a proposal distribution $Q$ with mean $\mu$
and covariance $\Sigma$ and giving each point the importance weight
$w_k \propto P(x_k)/Q(x_k)$, where $P(x) \propto e^{-\text{nllf}(x)}$ is the
posterior probability. The weighted sample follows the actual likelihood, so
it can be summarized with the same statistics and plots as an MCMC sample. The
weights are normalized by their sum, so constant offsets in nllf don't matter.

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

__all__ = ["Proposal", "ImportanceSample", "draw_within_bounds", "sample_from_cov"]

from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

DEFAULT_SAMPLES = 10000
"""Default number of points to draw from the covariance matrix."""

DEFAULT_NU = 10.0
"""Default degrees of freedom for the multivariate t proposal."""

LOW_ESS_FRACTION = 0.1
"""Warn when the effective sample size is below this fraction of the number of samples."""

NOT_AT_MINIMUM = 1.0
"""Warn that the fit is not at the minimum when a sampled point improves nllf by more than this."""

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
                " distribution. Check that the fit is at the minimum, and that every parameter"
                " affects the fit."
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


class ImportanceSample:
    r"""
    Likelihood-weighted sample of the posterior distribution, from :func:`sample_from_cov`.

    *proposal* is the distribution that the *points* were drawn from and
    *labels* are the parameter names. *logp* is $-\text{nllf}$ at each point,
    and *logq* is the log density of the proposal at each point. *acceptance*
    is the fraction of the points drawn from the proposal that were within the
    parameter bounds, *start_logp* is $-\text{nllf}$ at the center of the
    proposal, and *title* is the name of the fit problem.

    *weights* gives the importance weight of each point, scaled to have
    mean 1, and :attr:`ess` gives the effective sample size. Points where the
    model fails or the constraints are not met have zero weight.

    The sample can be used in place of the MCMC state from DREAM: :meth:`draw`
    returns a :class:`bumps.dream.state.Draw` that carries the weights, for use
    with :func:`bumps.dream.stats.var_stats` and the uncertainty plots.
    """

    def __init__(
        self,
        proposal: Proposal,
        labels: List[str],
        points: NDArray,
        logp: NDArray,
        logq: NDArray,
        acceptance: float = 1.0,
        start_logp: float = -np.inf,
        title: Optional[str] = None,
    ):
        self.proposal = proposal
        self.labels = list(labels)
        self.points = np.asarray(points, "d")
        self.logp = np.array(logp, "d")
        # Treat failed model evaluations as having zero probability.
        self.logp[np.isnan(self.logp)] = -np.inf
        self.logq = np.asarray(logq, "d")
        self.acceptance = acceptance
        self.start_logp = start_logp
        self.title = title
        log_w = self.logp - self.logq
        top = np.max(log_w, initial=-np.inf)
        if not np.isfinite(top):
            raise ValueError("None of the sampled points have a finite likelihood")
        w = np.exp(log_w - top)
        self.weights = w * (len(w) / np.sum(w))
        # Variable selection and derived variables, used by Draw as in MCMCDraw.
        self._good_chains = slice(None)
        self._shown = None
        self._integer_vars = None
        self._derived_fn = None
        self._derived_labels = None

    @property
    def Nsamples(self) -> int:
        """Number of points in the sample."""
        return len(self.logp)

    @property
    def ess(self) -> float:
        r"""Effective sample size $(\sum w)^2/\sum w^2$ of the importance weights."""
        w = self.weights
        return float(np.sum(w) ** 2 / np.sum(w**2))

    def summary(self) -> str:
        """Describe the sample and its effective sample size, with warnings if the result is unreliable."""
        n, ess = self.Nsamples, self.ess
        lines = [
            f"Uncertainty from {n} points drawn from the covariance matrix and weighted by the likelihood,"
            f" with effective sample size {ess:.0f} ({100 * ess / n:.1f}%)"
        ]
        if self.acceptance < 1:
            lines.append(
                f"{100 * (1 - self.acceptance):.1f}% of the points drawn were outside the parameter bounds and redrawn"
            )
        improvement = np.max(self.logp) - self.start_logp
        if np.isfinite(self.start_logp) and improvement > NOT_AT_MINIMUM:
            lines.append(
                f"Warning: a sampled point improves nllf by {improvement:.3g} over the best fit, so the fit"
                " is not at the minimum. Fit again before relying on these uncertainties."
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

    def show(self, portion: Optional[float] = None, figfile: Optional[str] = None):
        """Print the parameter statistics and plot the uncertainty, saving the plots if *figfile* is given."""
        from .dream.views import plot_all

        plot_all(self.draw(portion=portion), figfile=figfile)
        print(self.summary())

    def save(self, filename: str):
        """Write the weight, logp and parameter values of each point to *filename*-cov-sample.dat.gz."""
        header = "\t".join(["weight", "logp", *self.labels])
        data = np.column_stack((self.weights, self.logp, self.points))
        np.savetxt(f"{filename}-cov-sample.dat.gz", data, delimiter="\t", header=header)


def sample_from_cov(
    problem,
    samples: int = DEFAULT_SAMPLES,
    nu: float = DEFAULT_NU,
    x: Optional[NDArray] = None,
    cov: Optional[NDArray] = None,
    mapper: Optional[Callable[[NDArray], List[float]]] = None,
) -> ImportanceSample:
    """
    Draw a likelihood-weighted sample of the posterior around the best fit.

    *problem* is the fit problem, *x* is the best fit (default: the current
    parameter values) and *cov* is its covariance matrix (default:
    *problem.cov(x)*). *samples* is the number of points to evaluate, and *nu*
    is the degrees of freedom of the proposal distribution, or 0 for a normal
    distribution.

    *mapper* evaluates nllf at a set of points, such as the function returned by
    :meth:`bumps.mapper.MPMapper.start_mapper`. By default the points are
    evaluated one at a time in the current process. The problem parameters are
    left unchanged.
    """
    labels = problem.labels()
    if not labels:
        raise ValueError("There are no fitted parameters to sample")
    original = problem.getp()
    x = original if x is None else np.asarray(x, "d")
    try:
        problem.setp(x)
        start_logp = -problem.nllf()
        if cov is None:
            cov = problem.cov(x)
        proposal = Proposal(x, cov, nu)
        points, logq, drawn, accepted = draw_within_bounds(proposal, samples, problem.bounds(), 100 * samples)
        if not len(points):
            raise RuntimeError(
                f"None of the {drawn} points drawn from the covariance matrix were within the parameter"
                " bounds. Check that the fit is at the minimum, and that every parameter affects the fit."
            )
        nllf = mapper(points) if mapper is not None else [problem.nllf(p) for p in points]
    finally:
        problem.setp(original)
    sample = ImportanceSample(
        proposal,
        labels,
        points,
        -np.asarray(nllf, "d"),
        logq,
        acceptance=accepted / drawn,
        start_logp=start_logp,
        title=problem.name,
    )
    # TODO: Add derived/visible/integer variable support to the optimizers as well.
    fn, derived_labels = getattr(problem, "derive_vars", (None, None))
    if fn is not None:
        sample.set_derived_vars(fn, labels=derived_labels)
    visible_vars = getattr(problem, "visible_vars", None)
    if visible_vars is not None:
        sample.set_visible_vars(visible_vars)
    integer_vars = getattr(problem, "integer_vars", None)
    if integer_vars is not None:
        sample.set_integer_vars(integer_vars)
    return sample


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
    from .dream.stats import var_stats

    np.random.seed(2)
    # The posterior is N(m, C), but the proposal is off-center, 20% too narrow in a
    # and 20% too wide in b.
    m = np.array([1.0, 2.0])
    C = np.array([[0.04, 0.018], [0.018, 0.0225]])
    D = np.diag([0.8, 1.2])
    Cinv = np.linalg.inv(C)
    proposal = Proposal(m + [0.05, -0.03], D @ C @ D, nu=10)
    x, logq = proposal.draw(100_000)
    dx = x - m
    nllf = 0.5 * np.einsum("ij,jk,ik->i", dx, Cinv, dx)
    sample = ImportanceSample(proposal, ["a", "b"], x, -nllf, logq)
    assert 0.3 * len(x) < sample.ess < len(x)

    draw = sample.draw()
    assert np.allclose(np.average(draw.points, axis=0, weights=draw.weights), m, atol=0.005)
    assert np.allclose(np.cov(draw.points.T, aweights=draw.weights), C, rtol=0.05)
    for v, mk, sk in zip(var_stats(draw), m, np.sqrt(np.diag(C))):
        assert abs(v.mean - mk) < 0.03 * sk
        assert abs(v.std - sk) < 0.03 * sk
        assert np.allclose(v.p68, [mk - sk, mk + sk], atol=0.03 * sk)

    # Selections keep the weights aligned with the points.
    selected = sample.draw(selection={"a": (0.9, 1.1)})
    assert len(selected.weights) == len(selected.points) < len(draw.points)


def test_bounds():
    np.random.seed(3)
    # A half-normal posterior on x >= 0 sampled from a normal proposal truncated to the bounds.
    proposal = Proposal([0.0], [[1.0]], nu=0)
    x, logq, drawn, accepted = draw_within_bounds(proposal, 50_000, np.array([[0.0], [np.inf]]), max_draws=10**6)
    assert len(x) == 50_000 and np.all(x >= 0)
    assert abs(accepted / drawn - 0.5) < 0.01
    sample = ImportanceSample(proposal, ["x"], x, -0.5 * x[:, 0] ** 2, logq, acceptance=accepted / drawn)
    # The posterior matches the truncated proposal, so the weights are equal.
    assert np.allclose(sample.weights, 1.0)
    assert "outside the parameter bounds" in sample.summary()

    # Give up when the proposal never falls within the bounds.
    x, _, drawn, accepted = draw_within_bounds(proposal, 10, np.array([[50.0], [51.0]]), max_draws=1000)
    assert len(x) == 0 and drawn == 1000 and accepted == 0


def test_low_ess_warning():
    np.random.seed(4)
    # The proposal is narrow and centered two standard deviations from the posterior mean,
    # so the few points nearest the mean carry most of the weight.
    proposal = Proposal([2.0], [[0.25]], nu=0)
    x, logq = proposal.draw(1000)
    sample = ImportanceSample(proposal, ["x"], x, -0.5 * x[:, 0] ** 2, logq)
    assert sample.ess < LOW_ESS_FRACTION * len(x)
    assert "Warning" in sample.summary()


def test_sample_from_cov():
    from .curve import Curve
    from .dream.stats import var_stats
    from .errplot import error_points_from_state
    from .fitproblem import FitProblem
    from .fitters import fit
    from .lsqerror import stderr

    np.random.seed(5)
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
    sample = sample_from_cov(problem, samples=4000)
    assert np.array_equal(problem.getp(), best.x)
    # The model is linear, so the weighted sample matches the covariance matrix.
    assert sample.Nsamples == 4000 and sample.ess > 0.8 * sample.Nsamples
    dx = np.array([(v.p68[1] - v.p68[0]) / 2 for v in var_stats(sample.draw())])
    assert np.allclose(dx, stderr(problem.cov(best.x)), rtol=0.1)
    assert "Warning" not in sample.summary()

    # Model uncertainty plots draw points from the weighted sample.
    points = error_points_from_state(sample, nshown=20)
    assert points.shape == (20, 2)

    # Sampling away from the minimum is reported, and the parameters are left unchanged.
    offset = sample_from_cov(problem, samples=2000, x=best.x + 5 * best.dx)
    assert "not at the minimum" in offset.summary()
    assert np.array_equal(problem.getp(), best.x)
