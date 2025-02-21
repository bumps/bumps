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
differentiation on nllf.

:func:`jacobian_cov` takes the Jacobian and computes the covariance matrix $C$.
:func:`hessian_cov` takes the Hessian and computes the covariance matrix $C$.

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

    Note that the problem.residuals() method should not reuse memory for the
    returned value otherwise the derivative calculation (f(x+dx) - f(x))/dx
    will always be zero. The returned value need not be 1D, but it should be
    contiguous so that it can be reshaped to 1D without an extra copy. This
    will only be an issue for very large datasets.
    """
    p_init = problem.getp()
    if p is None:
        p = p_init
    p = np.asarray(p)
    bounds = getattr(problem, "bounds", lambda: None)()

    def f(p):
        problem.setp(p)
        # Return residuals as a vector even if f(x) returns a matrix otherwise
        # we cannot build a stacked Jacobian. We use reshape() rather than
        # flatten since this will avoid an unnecessary copy.
        return np.reshape(problem.residuals(), -1)

    J = _jacobian_forward(f, p, bounds, eps=step)
    problem.setp(p_init)
    return J


def _jacobian_forward(f, p, bounds, eps=None):
    n = len(p)
    # TODO: default to double precision epsilon
    step = 1e-4 if eps is None else np.sqrt(eps)

    # print("p",p,"step",step)
    h = abs(p) * step
    h[h == 0] = step
    if bounds is not None:
        h[h + p > bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    fx = f(p)  # Maybe fx.copy() to protect against reuse
    J = []
    for i in range(n):
        fx_plus = f(p + ee[i, :])
        J.append((fx_plus - fx) / h[i])
    return np.vstack(J).T


def _jacobian_central(f, p, bounds, eps=None):
    n = len(p)
    # TODO: default to double precision epsilon
    step = 1e-4 if eps is None else np.sqrt(eps)

    # print("p",p,"step",step)
    h = abs(p) * step
    h[h == 0] = step
    # if bounds is not None:
    #    h[h+p>bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    J = []
    for i in range(n):
        fx_minus = f(p - ee[i, :])  # Maybe fx.copy() to protect against reuse
        fx_plus = f(p + ee[i, :])
        J.append((fx_plus - fx_minus) / (2.0 * h[i]))
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
    bounds = getattr(problem, "bounds", lambda: None)()
    H = _hessian_forward(problem.nllf, p, bounds=bounds, eps=step)
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

    # print("p",p,"step",step)
    h = abs(p) * step
    h[h == 0] = step
    if bounds is not None:
        h[h + p > bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    g = np.empty(n, "d")
    for i in range(n):
        g[i] = f(p + ee[i, :])
    # print("fx",fx)
    # print("h",h, h[0])
    # print("g",g)
    H = np.empty((n, n), "d")
    for i in range(n):
        for j in range(i, n):
            fx_ij = f(p + ee[i, :] + ee[j, :])
            # print("fx_%d%d=%g"%(i,j,fx_ij))
            H[i, j] = (fx_ij - g[i] - g[j] + fx) / (h[i] * h[j])
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
    # step = np.sqrt(step)
    fx = f(p)

    h = abs(p) * step
    h[h == 0] = step
    # TODO: handle bounds on central difference formula
    # if bounds is not None:
    #    h[h+p>bounds[1]] *= -1.0  # step backward if forward step is out of bounds
    ee = np.diag(h)

    gp = np.empty(n, "d")
    gm = np.empty(n, "d")
    for i in range(n):
        gp[i] = f(p + ee[i, :])
        gm[i] = f(p - ee[i, :])
    H = np.empty((n, n), "d")
    for i in range(n):
        for j in range(i, n):
            fp_ij = f(p + ee[i, :] + ee[j, :])
            fm_ij = f(p - ee[i, :] - ee[j, :])
            # print("fx_%d%d=%g"%(i,j,fx_ij))
            H[i, j] = (fp_ij - gp[i] - gp[j] + fm_ij - gm[i] - gm[j] + 2.0 * fx) / (2.0 * h[i] * h[j])
            H[j, i] = H[i, j]
    return H


def perturbed_hessian(H, scale=None):
    """
    **DEPRECATED** Numerical testing has shown that the perturbed Hessian
    is too aggressive with its perturbation, and it is distorting the error
    too much, so use hessian_cov(H) instead.

    Adjust Hessian matrix to be positive definite.

    Returns the adjusted Hessian and its Cholesky decomposition.
    """
    from .quasinewton import modelhess

    n = H.shape[0]
    if scale is None:
        scale = np.ones(n)
    macheps = np.finfo("d").eps
    return modelhess(n, scale, macheps, H)


def chol_stderr(L):
    """
    Return parameter uncertainty from the Cholesky decomposition of the
    Hessian matrix, as returned, e.g., from the quasi-Newton optimizer BFGS
    or as calculated from :func:`perturbed_hessian` on the output of
    :func:`hessian` applied to the cost function problem.nllf.

    Note that this calls chol_cov to compute the inverse from the Cholesky
    decomposition, so use stderr(C) if you are already computing C = chol_cov().

    **Warning:** assumes H = L@L.T (numpy default) not H = U.T@U (scipy default).
    """
    # TODO: are there numerical tricks to get the diagonal without the full inv?
    return stderr(chol_cov(L))


def chol_cov(L):
    """
    Given the cholesky decomposition of the Hessian matrix H, compute
    the covariance matrix $C = H^{-1}$

    **Warning:** assumes H = L@L.T (numpy default) not H = U.T@U (scipy default).
    """
    Linv = np.linalg.inv(L)
    return np.dot(Linv.T.conj(), Linv)


def jacobian_cov(J, tol=1e-8):
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
    JTJinv = np.dot(vh.T.conj() / s**2, vh)
    return JTJinv


def hessian_cov(H, tol=1e-15):
    """
    Given Hessian H, return the covariance matrix inv(H).

    We provide some protection against singular matrices by setting
    singular values smaller than tolerance *tol* (relative to the largest
    singular value) to zero (see np.linalg.pinv for details).
    """
    # Find cov of f at p
    #     cov(f,p) = inv(H)
    # Use SVD
    #     H = U S V'
    #     inv(H) = inv(U S V')
    #            = inv(V') inv(S S) inv(U)
    #            = V inv(S S) U'
    #     J'J = (U S V')' (U S V')
    #         = V S' U' U S V'
    #         = V S S V'
    #     inv(J'J) = inv(V S S V')
    #              = inv(V') inv(S S) inv(V)
    #              = V inv (S S) V'
    return np.linalg.pinv(H, rcond=tol, hermitian=True)


def corr(C):
    """
    Convert covariance matrix $C$ to correlation matrix $R^2$.

    Uses $R = D^{-1} C D^{-1}$ where $D$ is the square root of the diagonal
    of the covariance matrix, or the standard error of each variable.
    """
    Dinv = 1.0 / stderr(C)
    return np.dot(Dinv, np.dot(C, Dinv))


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
    rosen = lambda x: (1.0 - x[0]) ** 2 + 105 * (x[1] - x[0] ** 2) ** 2
    p = np.array([1.0, 1.0])
    H = _hessian_forward(rosen, p, bounds=None, eps=1e-16)
    print("forward difference H", H)
    H = _hessian_central(rosen, p, bounds=None, eps=1e-16)
    print("central difference H", H)


def demo_jacobian():
    y = np.array([1.0, 2.0, 3.0])
    f = lambda x: x[0] * y + x[1]
    p = np.array([2.0, 3.0])
    J = _jacobian_forward(f, p, bounds=None, eps=1e-16)
    print("forward difference J", J)
    J = _jacobian_central(f, p, bounds=None, eps=1e-16)
    print("central difference J", J)


# https://en.wikipedia.org/wiki/Hilbert_matrix
# Note: 1-origin indices translated to 0-origin
def hilbert(n):
    """Generate ill-conditioned Hilbert matrix of size n x n"""
    return 1 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1)


# https://en.wikipedia.org/wiki/Hilbert_matrix#Properties
# Note: 1-origin indices translated to 0-origin
def hilbertinv(n):
    """Analytical inverse for ill-conditioned Hilbert matrix of size n x n"""
    Hinv = [
        [
            (-1) ** (i + j + 2) * (i + j + 1) * comb(n + i, n - j - 1) * comb(n + j, n - i - 1) * comb(i + j, i) ** 2
            for i in range(n)
        ]
        for j in range(n)
    ]
    return np.asarray(Hinv, dtype="d")


# From dheerosaur
# https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python/4941932#4941932
def comb(n, r):
    """n choose r combination function"""
    import operator as op
    from functools import reduce

    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def demo_stderr_hilbert(n=5):
    H = hilbert(n)
    C = hilbertinv(n)
    s = stderr(C)
    Hp, Lp = perturbed_hessian(H)
    Cp = chol_cov(Lp)
    sp = chol_stderr(Lp)
    Cdirect = hessian_cov(H)
    sdirect = stderr(Cdirect)
    with np.printoptions(precision=3):
        print("s ", s)
        print("sp", sp)
        print("sd", sdirect)
        print("R", corr(C))


def demo_stderr_perturbed():
    n = 5
    D = [1, 2, 3, 4, 5]
    # D = np.exp(10*np.random.rand(n)**2)
    D = [1e-3, 1e-2, 1e-1, 1, 10]

    D = np.asarray(D)
    L = np.tril(np.random.rand(n, n))
    np.fill_diagonal(L, D)
    H = np.dot(L, L.T)
    Hp, Lp = perturbed_hessian(H)
    C = chol_cov(Lp)
    s = chol_stderr(Lp)

    from scipy.linalg import inv

    Ldirect = np.linalg.cholesky(H)
    Cdirect = inv(H)
    Cp = inv(Hp)

    sdirect = np.sqrt(np.diag(Cdirect))
    sp = np.sqrt(np.diag(Cp))
    sdirect_chol = chol_stderr(Ldirect)

    parts = dict(
        L_original=L,
        L_direct=Ldirect,
        L_perturbed=Lp,
        # H=H,
        # H_perturbed=Hp,
        # C_direct=Cdirect,
        # C_from_Hp=Cp,
        # C_from_Lp=C,
    )
    with np.printoptions(precision=3):
        print("%20s" % ("perturbation"), hp[0, 0] - h[0, 0])
        for k, v in parts.items():
            print("%20s" % (k + " diag"), np.diag(v))
        # print("eigc", list(sorted(np.linalg.eigvals(c))))
        # print("eigcp", list(sorted(np.linalg.eigvals(cp))))
        # print("eigh", list(sorted(1/np.linalg.eigvals(h))))
        # print("eighp", list(sorted(1/np.linalg.eigvals(hp))))
        print("h cond     ", np.linalg.cond(h))
        print("rel err dc ", abs((c - cdirect) / cdirect).max())
        print("de         ", sp - s)
        print("s direct   ", sdirect)
        print("s chol     ", sdirect_chol)
        print("s perturbed", sp)
        print("s          ", s)
        print("rel err ds ", abs((s - sdirect) / sdirect).max())
        print("unperturbed ds", abs((sdirect_chol - sdirect) / sdirect).max())


if __name__ == "__main__":
    # demo_hessian()
    # demo_jacobian()
    # demo_stderr_perturbed()
    demo_stderr_hilbert(10)
