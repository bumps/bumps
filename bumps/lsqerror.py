r"""
Least squares error analysis.

Given a data set with gaussian uncertainty on the points, and a model which
is differentiable at the minimum, the parameter uncertainty can be estimated
from the covariance matrix at the minimum.  The model and data are wrapped in
a problem object, which must define the following methods:

    ============ ============================================
    getp()       get the current value of the model
    setp(p)      set a new value in the model
    nllf(p)      negative log likelihood function
    residuals(p) residuals around its current value
    bounds()     get the bounds on the parameter p [optional]
    ============ ============================================

:func:`jacobian` computes the Jacobian matrix $J$ using numerical
differentiation on residuals. Derivatives are computed using the center
point formula, with two evaluations per dimension.  If the problem has
analytic derivatives with respect to the fitting parameters available,
then these should be used to compute the Jacobian instead.

:func:`hessian` computes the Hessian matrix $H$ using numerical
differentiation on nllf.  This uses the center point formula, with
two evaluations for each (i,j) combination.

:func:`cov` takes the Jacobian and computes the covariance matrix $C$.

:func:`corr` uses the off-diagonal elements of $C$ to compute correlation
coefficients $R^2_{ij}$ between the parameters.

:func:`stderr` computes the uncertain $\sigma_i$ from covariance matrix $C$,
assuming that the $C_\text{diag}$ contains $\sigma_i^2$, which should be
the case for functions which are approximately linear near the minimum.

:func:`max_correlation` takes $R^2$ and returns the maximum correlation.

The user should be shown the uncertainty $\sigma_i$ for each parameter,
and if there are strong parameter correlations (e.g., $R^2_\text{max} > 0.2$),
the correlation matrix as well.

The bounds method for the problem is optional, and is used only to determine
the step size needed for the numerical derivative.  If bounds are not present
and finite, the current value for the parameter is used as a basis to
estimate step size.

"""

import numpy as np
from . import numdifftools as nd


def jacobian(problem, p=None, step=None):
    """
    Returns the derivative wrt the fit parameters at point p.

    Numeric derivatives are calculated based on step, where step is
    the portion of the total range for parameter j, or the portion of
    point value p_j if the range on parameter j is infinite.
    """
    p_init = problem.getp()
    if p is None:
        p = p_init
    p = np.asarray(p)
    J = nd.Jacobian(problem.residuals)(p)
    problem.setp(p_init)
    return J


def hessian(problem, p=None, step=None):
    """
    Returns the derivative wrt to the fit parameters at point p.
    """
    p_init = problem.getp()
    if p is None:
        p = p_init
    p = np.asarray(p)
    import numdifftools as nd
    H = nd.Hessian(problem.nllf)(p)
    #bounds = getattr(problem, 'bounds', lambda: None)()
    #H2 = _simple_hessian(problem.nllf, p, step=step, bounds=bounds)
    # print(H-H2)
    problem.setp(p_init)
    return H


def hessian_diag(problem, p=None, step=None):
    """
    Returns the derivative wrt to the fit parameters at point p.
    """
    p_init = problem.getp()
    if p is None:
        p = p_init
    p = np.asarray(p)
    H = nd.Hessdiag(problem.nllf)(p)
    #bounds = getattr(problem, 'bounds', lambda: None)()
    #H2 = _simple_hessian(problem.nllf, p, step=step, bounds=bounds)
    # print(H-H2)
    problem.setp(p_init)
    return H

def _delta(p, bounds, step):
    if step is None:
        step = 1e-8
    if bounds is not None:
        lo, hi = bounds
        delta = (hi - lo) * step
        # For infinite ranges, use p*1e-8 for the step size
        idx = np.isinf(delta)
        # print "J",idx,delta,p,type(idx),type(delta),type(p)
        delta[idx] = p[idx] * step
    else:
        delta = p * step
    delta[delta == 0] = step
    return delta


def perturbed_hessian(H, scale=None):
    """
    Adjust Hessian matrix to be positive definite.

    Returns the adjusted Hessian and its Cholesky decomposition.
    """
    from .quasinewton import modelhess
    n = H.shape[0]
    if scale is None:
        scale = np.ones(n)
    macheps = np.finfo('d').eps
    return modelhess(n, scale, macheps, H)


def chol_stderr(L):
    """
    Return parameter uncertainty from the Cholesky decomposition of the
    Hessian matrix, as returned, e.g., from the quasi-Newton optimizer BFGS
    or as calculated from :func:`perturbed_hessian` on the output of
    :func:`hessian` applied to the cost function problem.nllf.
    """
    return np.sqrt(1. / np.diag(L))


def chol_cov(L):
    """
    Given the cholesky decomposition of the Hessian matrix H, compute
    the covariance matrix $C = H^{-1}$
    """
    Linv = np.linalg.inv(L)
    return np.dot(Linv.T.conj(), Linv)


def cov(J, tol=1e-8):
    """
    Given Jacobian J, return the covariance matrix inv(J'J).

    We provide some protection against singular matrices by setting
    singular values smaller than tolerance *tol* to the tolerance
    value.
    """

    # Find cov of f at p
    #     cov(f,p) = inv(J'J)
    # Use SVD
    #     J = U S V'
    #     J'J = (U S V')' (U S V')
    #         = V S' U' U S V'
    #         = V S S V'
    #     inv(J'J) = inv(V S S V')
    #              = inv(V') inv(S S) inv(V)
    #              = V inv (S S) V'
    u, s, vh = np.linalg.svd(J, 0)
    s[s <= tol] = tol
    JTJinv = np.dot(vh.T.conj() / s ** 2, vh)
    return JTJinv


def corr(C):
    """
    Convert covariance matrix $C$ to correlation matrix $R^2$.

    Uses $R = D^{-1} C D^{-1}$ where $D$ is the square root of the diagonal
    of the covariance matrix, or the standard error of each variable.
    """
    Dinv = 1. / stderr(cov)
    return np.dot(Dinv, np.dot(cov, Dinv))


def max_correlation(Rsq):
    """
    Return the maximum correlation coefficient for any pair of variables
    in correlation matrix Rsq.
    """
    return np.max(np.tril(Rsq, k=-1))


def stderr(C):
    r"""
    Return parameter uncertainty from the covariance matrix C.

    This is just the square root of the diagonal, without any correction
    for covariance.

    If measurement uncertainty is unknown, scale the returned uncertainties
    by $\sqrt{\chi^2_N}$, where $\chi^2_N$ is the sum squared residuals
    divided by the degrees  of freedom.  This will match the uncertainty on
    the parameters to the observed scatter assuming the model is correct and
    the fit is optimal.  This will also be appropriate for weighted fits
    when the true measurement uncertainty dy_i is known up to a scaling
    constant for all y_i.

    Standard error on scipy.optimize.curve_fit always includes the chisq
    correction, whereas scipy.optimize.leastsq never does.
    """
    return np.sqrt(np.diag(C))
