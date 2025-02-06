# This program is public domain
"""
BSpline calculator.

Given a set of knots, compute the cubic B-spline interpolation.
"""

__all__ = ["bspline", "pbs"]

import numpy as np
from numpy import maximum as max, minimum as min


def pbs(x, y, t, clamp=True, parametric=True):
    """
    Evaluate the parametric B-spline px(t),py(t).

    *x* and *y* are the control points, and *t* are the points
    in [0,1] at which they are evaluated.   The *x* values are
    sorted so that the spline describes a function.

    The spline goes through the control points at the ends. If *clamp*
    is True, the derivative of the spline at both ends is zero. If *clamp*
    is False, the derivative at the ends is equal to the slope connecting
    the final pair of control points.

    If *parametric* is False, then parametric points t' are chosen such
    that x(t') = *t*.

    The B-spline knots are chosen to be equally spaced within [0,1].
    """
    x = list(sorted(x))
    knot = np.hstack((0, 0, np.linspace(0, 1, len(y)), 1, 1))
    cx = np.hstack((x[0], x[0], x[0], (2 * x[0] + x[1]) / 3, x[1:-1], (2 * x[-1] + x[-2]) / 3, x[-1]))
    if clamp:
        cy = np.hstack((y[0], y[0], y[0], y, y[-1]))
    else:
        cy = np.hstack((y[0], y[0], y[0], y[0] + (y[1] - y[0]) / 3, y[1:-1], y[-1] + (y[-2] - y[-1]) / 3, y[-1]))

    if parametric:
        return _bspline3(knot, cx, t), _bspline3(knot, cy, t)

    # Find parametric t values corresponding to given z values
    # First try a few newton steps
    xt = np.interp(t, x, np.linspace(0, 1, len(x)))
    with np.errstate(all="ignore"):
        for _ in range(6):
            pt, dpt = _bspline3(knot, cx, xt, nderiv=1)
            xt -= (pt - t) / dpt
        idx = np.isnan(xt) | (abs(_bspline3(knot, cx, xt) - t) > 1e-9)

    # Use bisection when newton fails
    if idx.any():
        missing = t[idx]
        # print missing
        t_lo, t_hi = 0 * missing, 1 * missing
        for _ in range(30):  # bisection with about 1e-9 tolerance
            trial = (t_lo + t_hi) / 2
            ptrial = _bspline3(knot, cx, trial)
            tidx = ptrial < missing
            t_lo[tidx] = trial[tidx]
            t_hi[~tidx] = trial[~tidx]
        xt[idx] = (t_lo + t_hi) / 2
    # print "err",np.max(abs(_bspline3(knot,cx,t)-xt))

    # Return y evaluated at the interpolation points
    return _bspline3(knot, cx, xt), _bspline3(knot, cy, xt)


def bspline(y, xt, clamp=True):
    """
    Evaluate the B-spline with control points *y* at positions *xt* in [0,1].

    The spline goes through the control points at the ends.  If *clamp*
    is True, the derivative of the spline at both ends is zero.  If *clamp*
    is False, the derivative at the ends is equal to the slope connecting
    the final pair of control points.

    B-spline knots are chosen to be equally spaced within [0,1].
    """
    knot = np.hstack((0, 0, np.linspace(0, 1, len(y)), 1, 1))
    if clamp:
        cy = np.hstack(([y[0]] * 3, y, y[-1]))
    else:
        cy = np.hstack((y[0], y[0], y[0], y[0] + (y[1] - y[0]) / 3, y[1:-1], y[-1] + (y[-2] - y[-1]) / 3, y[-1]))
    return _bspline3(knot, cy, xt)


def _bspline3(knot, control, t, nderiv=0):
    """
    Evaluate the B-spline specified by the given *knot* sequence and
    *control* values at the parametric points *t*.  *nderiv* selects
    the function or derivative to evaluate.
    """
    knot, control, t = [np.asarray(v) for v in (knot, control, t)]

    # Deal with values outside the range
    valid = (t > knot[0]) & (t <= knot[-1])
    tv = t[valid]
    f = np.zeros(t.shape)
    f[t <= knot[0]] = control[0]
    f[t >= knot[-1]] = control[-1]

    # Find B-Spline parameters for the individual segments
    end = len(knot) - 1
    segment = knot.searchsorted(tv) - 1
    tm2 = knot[max(segment - 2, 0)]
    tm1 = knot[max(segment - 1, 0)]
    tm0 = knot[max(segment - 0, 0)]
    tp1 = knot[min(segment + 1, end)]
    tp2 = knot[min(segment + 2, end)]
    tp3 = knot[min(segment + 3, end)]

    p4 = control[min(segment + 3, end)]
    p3 = control[min(segment + 2, end)]
    p2 = control[min(segment + 1, end)]
    p1 = control[min(segment + 0, end)]

    # Compute second and third derivatives.
    if nderiv > 1:
        # Normally we require a recursion for Q, R and S to compute
        # df, d2f and d3f respectively, however Q can be computed directly
        # from intermediate values of P, S has a recursion of depth 0,
        # which leaves only the R recursion of depth 1 in the calculation
        # below.
        q4 = (p4 - p3) * 3 / (tp3 - tm0)
        q3 = (p3 - p2) * 3 / (tp2 - tm1)
        q2 = (p2 - p1) * 3 / (tp1 - tm2)
        r4 = (q4 - q3) * 2 / (tp2 - tm0)
        r3 = (q3 - q2) * 2 / (tp1 - tm1)
        if nderiv > 2:
            s4 = (r4 - r3) / (tp1 - tm0)
            d3f = np.zeros(t.shape)
            d3f[valid] = s4
        r4 = ((tv - tm0) * r4 + (tp1 - tv) * r3) / (tp1 - tm0)
        d2f = np.zeros(t.shape)
        d2f[valid] = r4

    # Compute function value and first derivative
    p4 = ((tv - tm0) * p4 + (tp3 - tv) * p3) / (tp3 - tm0)
    p3 = ((tv - tm1) * p3 + (tp2 - tv) * p2) / (tp2 - tm1)
    p2 = ((tv - tm2) * p2 + (tp1 - tv) * p1) / (tp1 - tm2)
    p4 = ((tv - tm0) * p4 + (tp2 - tv) * p3) / (tp2 - tm0)
    p3 = ((tv - tm1) * p3 + (tp1 - tv) * p2) / (tp1 - tm1)
    if nderiv >= 1:
        df = np.zeros(t.shape)
        df[valid] = (p4 - p3) * 3 / (tp1 - tm0)
    p4 = ((tv - tm0) * p4 + (tp1 - tv) * p3) / (tp1 - tm0)
    f[valid] = p4

    if nderiv == 0:
        return f
    elif nderiv == 1:
        return f, df
    elif nderiv == 2:
        return f, df, d2f
    else:
        return f, df, d2f, d3f


"""
def bspline_control(y, clamp=True):
    return _find_control(y, clamp=clamp)


def pbs_control(x, y, clamp=True):
    return _find_control(x, clamp=clamp), _find_control(y, clamp=clamp)


def _find_control(v, clamp=True):
    raise NotImplementedError("B-spline interpolation doesn't work yet")
    from scipy.linalg import solve_banded
    n = len(v)
    udiag = np.hstack([0, 0, 0, [1 / 6] * (n - 3), 0.25, 0.3])
    ldiag = np.hstack([-0.3, 0.25, [1 / 6] * (n - 3), 0, 0, 0])
    mdiag = np.hstack([1, 0.3, 7 / 12, [2 / 3] * (n - 4), 7 / 12, -0.3, 1])
    A = np.vstack([ldiag, mdiag, udiag])
    if clamp:
        # First derivative is zero at ends
        bl, br = 0, 0
    else:
        # First derivative at ends follows line between final control points
        bl, br = (v[1] - v[0]) * n, (v[-1] - v[-2]) * n
    b = np.hstack([v[0], bl, v[1:n - 1], br, v[-1]])
    x = solve_banded((1, 1), A, b)
    return x  # x[1:-1]
"""

# ===========================================================================
# test code


def speed_check():
    """
    Print the time to evaluate 400 points on a 7 knot spline.
    """
    import time

    x = np.linspace(0, 1, 7)
    x[1], x[-2] = x[2], x[-3]
    y = [9, 11, 2, 3, 8, 0, 2]
    t = np.linspace(0, 1, 400)
    t0 = time.time()
    for _ in range(1000):
        bspline(y, t, clamp=True)
    print("bspline (ms)", (time.time() - t0) / 1000)


def _check(expected, got, tol):
    """
    Check that value matches expected within tolerance.

    If *expected* is never zero, use relative error for tolerance.
    """
    relative = (np.isscalar(expected) and expected != 0) or (not np.isscalar(expected) and all(expected != 0))
    if relative:
        norm = np.linalg.norm((expected - got) / expected)
    else:
        norm = np.linalg.norm(expected - got)
    if norm >= tol:
        msg = [
            "expected %s" % str(expected),
            "got %s" % str(got),
            "tol %s norm %s" % (tol, norm),
        ]
        raise ValueError("\n".join(msg))


def _derivs(x, y):
    """
    Compute numerical derivative for a function evaluated on a fine grid.
    """
    # difference formula
    return (y[1] - y[0]) / (x[1] - x[0]), (y[-1] - y[-2]) / (x[-1] - x[-2])
    # 5-point difference formula
    # left = (y[0]-8*y[1]+8*y[3]-y[4]) / 12 / (x[1]-x[0])
    # right = (y[-5]-8*y[-4]+8*y[-2]-y[-1]) / 12 / (x[-1]-x[-2])
    # return left, right


def test():
    """bspline tests"""
    h = 1e-10
    t = np.linspace(0, 1, 100)
    dt = np.array([0, h, 2 * h, 3 * h, 4 * h, 1 - 4 * h, 1 - 3 * h, 1 - 2 * h, 1 - h, 1])
    y = [9, 11, 2, 3, 8, 0, 2]
    n = len(y)
    xeq = np.linspace(0, 1, n)
    x = xeq + 0
    x[0], x[-1] = (x[0] + x[1]) / 2, (x[-2] + x[-1]) / 2
    dx = np.array(
        [
            x[0],
            x[0] + h,
            x[0] + 2 * h,
            x[0] + 3 * h,
            x[0] + 4 * h,
            x[-1] - 4 * h,
            x[-1] - 3 * h,
            x[-1] - 2 * h,
            x[-1] - h,
            x[-1],
        ]
    )

    # ==== Check that bspline mat:w
    # ches pbs with equally spaced x

    yt = bspline(y, t, clamp=True)
    xtp, ytp = pbs(xeq, y, t, clamp=True, parametric=False)
    _check(t, xtp, 1e-8)
    _check(yt, ytp, 1e-8)

    xtp, ytp = pbs(xeq, y, t, clamp=True, parametric=True)
    _check(t, xtp, 1e-8)
    _check(yt, ytp, 1e-8)

    yt = bspline(y, t, clamp=False)
    xtp, ytp = pbs(xeq, y, t, clamp=False, parametric=False)
    _check(t, xtp, 1e-8)
    _check(yt, ytp, 1e-8)

    xtp, ytp = pbs(xeq, y, t, clamp=False, parametric=True)
    _check(t, xtp, 1e-8)
    _check(yt, ytp, 1e-8)

    # ==== Check bspline f at end points

    yt = bspline(y, t, clamp=True)
    _check(y[0], yt[0], 1e-12)
    _check(y[-1], yt[-1], 1e-12)

    yt = bspline(y, t, clamp=False)
    _check(y[0], yt[0], 1e-12)
    _check(y[-1], yt[-1], 1e-12)

    xt, yt = pbs(x, y, t, clamp=True, parametric=False)
    _check(x[0], xt[0], 1e-8)
    _check(x[-1], xt[-1], 1e-8)
    _check(y[0], yt[0], 1e-8)
    _check(y[-1], yt[-1], 1e-8)

    xt, yt = pbs(x, y, t, clamp=True, parametric=True)
    _check(x[0], xt[0], 1e-8)
    _check(x[-1], xt[-1], 1e-8)
    _check(y[0], yt[0], 1e-8)
    _check(y[-1], yt[-1], 1e-8)

    xt, yt = pbs(x, y, t, clamp=False, parametric=False)
    _check(x[0], xt[0], 1e-8)
    _check(x[-1], xt[-1], 1e-8)
    _check(y[0], yt[0], 1e-8)
    _check(y[-1], yt[-1], 1e-8)

    xt, yt = pbs(x, y, t, clamp=False, parametric=True)
    _check(x[0], xt[0], 1e-8)
    _check(x[-1], xt[-1], 1e-8)
    _check(y[0], yt[0], 1e-8)
    _check(y[-1], yt[-1], 1e-8)

    # ==== Check f' at end points
    yt = bspline(y, dt, clamp=True)
    left, right = _derivs(dt, yt)
    _check(0, left, 1e-8)
    _check(0, right, 1e-8)

    xt, yt = pbs(x, y, dx, clamp=True, parametric=False)
    left, right = _derivs(xt, yt)
    _check(0, left, 1e-8)
    _check(0, right, 1e-8)

    xt, yt = pbs(x, y, dt, clamp=True, parametric=True)
    left, right = _derivs(xt, yt)
    _check(0, left, 1e-8)
    _check(0, right, 1e-8)

    yt = bspline(y, dt, clamp=False)
    left, right = _derivs(dt, yt)
    _check((y[1] - y[0]) * (n - 1), left, 5e-4)
    _check((y[-1] - y[-2]) * (n - 1), right, 5e-4)

    xt, yt = pbs(x, y, dx, clamp=False, parametric=False)
    left, right = _derivs(xt, yt)
    _check((y[1] - y[0]) / (x[1] - x[0]), left, 5e-4)
    _check((y[-1] - y[-2]) / (x[-1] - x[-2]), right, 5e-4)

    xt, yt = pbs(x, y, dt, clamp=False, parametric=True)
    left, right = _derivs(xt, yt)
    _check((y[1] - y[0]) / (x[1] - x[0]), left, 5e-4)
    _check((y[-1] - y[-2]) / (x[-1] - x[-2]), right, 5e-4)

    # ==== Check interpolator
    # yc = bspline_control(y)
    # print("y",y)
    # print("p(yc)",bspline(yc,xeq))


def demo():
    """Show bsline curve for a set of control points."""
    from pylab import linspace, subplot, plot, legend, show

    # y = [9 ,6, 1, 3, 8, 4, 2]
    # y = [9, 11, 13, 3, -2, 0, 2]
    y = [9, 11, 2, 3, 8, 0]
    # y = [9, 9, 1, 3, 8, 2, 2]
    x = linspace(0, 1, len(y))
    t = linspace(x[0], x[-1], 400)
    subplot(211)
    plot(t, bspline(y, t, clamp=False), "-.y", label="unclamped bspline")  # bspline
    # bspline
    plot(t, bspline(y, t, clamp=True), "-y", label="clamped bspline")
    plot(sorted(x), y, ":oy", label="control points")
    legend()
    # left, right = _derivs(t, bspline(y, t, clamp=False))
    # print(left, (y[1] - y[0]) / (x[1] - x[0]))

    subplot(212)
    xt, yt = pbs(x, y, t, clamp=False)
    plot(xt, yt, "-.b", label="unclamped pbs")  # pbs
    xt, yt = pbs(x, y, t, clamp=True)
    plot(xt, yt, "-b", label="clamped pbs")  # pbs
    # xt,yt = pbs(x,y,t,clamp=True, parametric=True)
    # plot(xt,yt,'-g') # pbs
    plot(sorted(x), y, ":ob", label="control points")
    legend()
    show()


# B-Spline control point inverse function is not yet implemented
"""
def demo_interp():
    from pylab import linspace, plot, show
    x = linspace(0, 1, 7)
    y = [9, 11, 2, 3, 8, 0, 2]
    t = linspace(0, 1, 400)
    yc = bspline_control(y, clamp=True)
    xc = linspace(x[0], x[-1], 9)
    plot(xc, yc, ':oy', x, y, 'xg')
    #knot = np.hstack((0, np.linspace(0,1,len(y)), 1))
    #fy = _bspline3(knot,yc,t)
    fy = bspline(yc, t, clamp=True)
    plot(t, fy, '-.y')
    show()
"""

if __name__ == "__main__":
    # test()
    demo()
    # demo_interp()
    # speed_check()
