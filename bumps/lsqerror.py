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
from __future__ import print_function

import numpy as np
#from . import numdifftools as nd
#import numdifftools as nd

def gradient(problem, p=None, step=None):
    r = problem.residuals()
    J = jacobian(problem, p=p, step=step)
    return np.dot(J.T, r)

# TODO: restructure lsqerror to use mapper for evaluating multiple f
# doesn't work for jacobian since mapper returns nllf; would need to
# expand mapper to implement a variety of different functions.
def jacobian(problem, p=None, step=None):
    """
    Returns the derivative wrt the fit parameters at point p.

    Numeric derivatives are calculated based on step, where step is
    the portion of the total range for parameter j, or the portion of
    point value p_j if the range on parameter j is infinite.

    The current point is preserved.
    """
    p_init = problem.getp()
    if p is None:
        p = p_init
    p = np.asarray(p)
    bounds = getattr(problem, 'bounds', lambda: None)()
    def f(p):
        problem.setp(p)
        return problem.residuals()
    J = _jacobian_forward(f, p, bounds, eps=step)
    #J = nd.Jacobian(problem.residuals)(p)
    problem.setp(p_init)
    return J

def _jacobian_forward(f, p, bounds, eps=None):
    n = len(p)
    # TODO: default to double precision epsilon
    step = 1e-4 if eps is None else np.sqrt(eps)
    fx = f(p)

    #print("p",p,"step",step)
    h = abs(p)*step
    h[h == 0] = step
    if bounds is not None:
        h[h+p > bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    J = []
    for i in range(n):
        J.append((f(p + ee[i, :]) - fx)/h[i])
    return np.vstack(J).T

def _jacobian_central(f, p, bounds, eps=None):
    n = len(p)
    # TODO: default to double precision epsilon
    step = 1e-4 if eps is None else np.sqrt(eps)

    #print("p",p,"step",step)
    h = abs(p)*step
    h[h == 0] = step
    #if bounds is not None:
    #    h[h+p>bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    J = []
    for i in range(n):
        J.append((f(p + ee[i, :]) - f(p - ee[i, :])) / (2.0*h[i]))
    return np.vstack(J).T


def hessian(problem, p=None, step=None):
    """
    Returns the derivative wrt to the fit parameters at point p.

    The current point is preserved.
    """
    p_init = problem.getp()
    if p is None:
        p = p_init
    p = np.asarray(p)
    bounds = getattr(problem, 'bounds', lambda: None)()
    H = _hessian_forward(problem.nllf, p, bounds=bounds, eps=step)
    #H = nd.Hessian(problem.nllf)(p)
    #print("Hessian",H)
    problem.setp(p_init)
    return H

def _hessian_forward(f, p, bounds, eps=None):
    # type: (Callable[[np.ndarray], float], np.ndarray, Optional[np.ndarray]) -> np.ndarray
    """
    Forward difference Hessian.
    """
    n = len(p)
    # TODO: default to double precision epsilon
    step = 1e-4 if eps is None else np.sqrt(eps)
    fx = f(p)

    #print("p",p,"step",step)
    h = abs(p)*step
    h[h == 0] = step
    if bounds is not None:
        h[h+p > bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    g = np.empty(n, 'd')
    for i in range(n):
        g[i] = f(p + ee[i, :])
    #print("fx",fx)
    #print("h",h, h[0])
    #print("g",g)
    H = np.empty((n, n), 'd')
    for i in range(n):
        for j in range(i, n):
            fx_ij = f(p + ee[i, :] + ee[j, :])
            #print("fx_%d%d=%g"%(i,j,fx_ij))
            H[i, j] = (fx_ij - g[i] - g[j] + fx) / (h[i]*h[j])
            H[j, i] = H[i, j]
    return H

def _hessian_central(f, p, bounds, eps=None):
    # type: (Callable[[np.ndarray], float], np.ndarray, Optional[np.ndarray]) -> np.ndarray
    """
    Central difference Hessian.
    """
    n = len(p)
    # TODO: default to double precision epsilon
    step = 1e-4 if eps is None else np.sqrt(eps)
    #step = np.sqrt(step)
    fx = f(p)

    h = abs(p)*step
    h[h == 0] = step
    # TODO: handle bounds on central difference formula
    #if bounds is not None:
    #    h[h+p>bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    gp = np.empty(n, 'd')
    gm = np.empty(n, 'd')
    for i in range(n):
        gp[i] = f(p + ee[i, :])
        gm[i] = f(p - ee[i, :])
    H = np.empty((n, n), 'd')
    for i in range(n):
        for j in range(i, n):
            fp_ij = f(p + ee[i, :] + ee[j, :])
            fm_ij = f(p - ee[i, :] - ee[j, :])
            #print("fx_%d%d=%g"%(i,j,fx_ij))
            H[i, j] = (fp_ij - gp[i] - gp[j] + fm_ij - gm[i] - gm[j] + 2.0*fx) / (2.0*h[i]*h[j])
            H[j, i] = H[i,j]
    return H


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


def demo_hessian():
    rosen = lambda x: (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    p = np.array([1., 1.])
    H = _hessian_forward(rosen, p, bounds=None, eps=1e-16)
    print("forward difference H", H)
    H = _hessian_central(rosen, p, bounds=None, eps=1e-16)
    print("central difference H", H)

    #from . import numdifftools as nd
    #import numdifftools as nd
    #Hfun = nd.Hessian(rosen)
    #print("numdifftools H", Hfun(p))

def demo_jacobian():
    y = np.array([1., 2., 3.])
    f = lambda x: x[0]*y + x[1]
    p = np.array([2., 3.])
    J = _jacobian_forward(f, p, bounds=None, eps=1e-16)
    print("forward difference J", J)
    J = _jacobian_central(f, p, bounds=None, eps=1e-16)
    print("central difference J", J)

    #from . import numdifftools as nd
    #import numdifftools as nd
    #Jfun = nd.Jacobian(f)
    #print("numdifftools J", Jfun(p))

if __name__ == "__main__":
    demo_hessian()
    demo_jacobian()
