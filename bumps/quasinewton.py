# Copyright (C) 2009-2010, University of Maryland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Author: Ismet Sahin
"""
BFGS quasi-newton optimizer.

All modules in this file are implemented from the book
"Numerical Methods for Unconstrained Optimization and Nonlinear Equations" by
J.E. Dennis and Robert B. Schnabel (Only a few minor modifications are done).

The interface is through the :func:`quasinewton` function.  Here is an
example call::

    n = 2
    x0 = [-0.9 0.9]'
    fn = lambda p: (1-p[0])**2 + 100*(p[1]-p[0]**2)**2
    grad = lambda p: array([-2*(1-p[0]) - 400*(p[1]-p[0]**2)*p[0], 200*p[1]])
    Sx = ones(n,1)
    typf = 1                       # todo. see what default value is the best
    macheps = eps
    eta = eps
    maxstep = 100
    gradtol = 1e-6
    steptol = 1e-12                # do not let steptol larger than 1e-9
    itnlimit = 1000
    result = quasinewton(fn, x0, grad, Sx, typf,
                         macheps, eta, maxstep, gradtola, steptol, itnlimit)
    print("status code %d"%result['status'])
    print("x_min=%s, f(x_min)=%g"%(str(result['x']),result['fx']))
    print("iterations, function calls, linesearch function calls",
          result['iterations'],result['evals'],result['linesearch_evals'])
"""
from __future__ import print_function

__all__ = ["quasinewton"]

from numpy import inf, sqrt, isnan, isinf, finfo, diag, zeros, ones
from numpy import array, linalg, inner, outer, dot, amax, maximum

STATUS = {
    1: "Gradient < tolerance",
    2: "Step size < tolerance",
    3: "Invalid point in line search",
    4: "Iterations exceeded",
    5: "Max step taken --- function unbounded?",
    6: "User abort",
    7: "Iterations exceeded in line search",
    8: "Line search step size is too small",
    9: "Singular Hessian",
}


def quasinewton(fn, x0=None, grad=None, Sx=None, typf=1, macheps=None, eta=None,
                maxstep=100, gradtol=1e-6, steptol=1e-12, itnlimit=2000,
                abort_test=None, monitor=lambda **kw: True):
    r"""
    Run a quasinewton optimization on the problem.

    *fn(x)* is the cost function, which takes a point x and returns a scalar fx.

    *x0* is the initial point

    *grad* is the analytic gradient (if available)

    *Sx* is a scale vector indicating the typical values for parameters in
    the fitted result. This is used for a variety of things such as setting
    the step size in the finite difference approximation to the gradient, and
    controlling numerical accuracy in calculating the Hessian matrix.  If for
    example some of your model parameters are in the order of 1e-6, then Sx
    for those parameters should be set to 1e-6. Default: [1, ...]

    *typf* is the typical value for f(x) near the minimum.  This is used along
    with gradtol to check the gradient stopping condition.  Default: 1

    *macheps* is the minimum value that can be added to 1 to produce a number
    not equal to 1.  Default: numpy.finfo(float).eps

    *eta* adapts the numerical gradient calculations to machine precision.
    Default: *macheps*

    *maxstep* is the maximum step size in any gradient step, after normalizing
    by *Sx*. Default: 100

    *gradtol* is a stopping condition for the fit based on the amount of
    improvement expected at the next step.  Default: 1e-6

    *steptol* is a stopping condition for the fit based on the size
    of the step. Default: 1e-12

    *itnlimit* is the maximum number of steps to take before stopping.
    Default: 2000

    *abort_test* is a function which tests whether the user has requested
    abort. Default: None.

    *monitor(x,fx,step)* is called every iteration so that a user interface
    function can monitor the progress of the fit.  Default: lambda \*\*kw: True


    Returns the fit result as a dictionary:

    *status* is a status code indicating why the fit terminated.  Turn the
    status code into a string with *STATUS[result.status]*.  Status values
    vary from 1 to 9, with 1 and 2 indicating convergence and the remaining
    codes indicating some form of premature termination.

    *x* is the minimum point

    *fx* is the value fn(x) at the minimum

    *H* is the approximate Hessian matrix, which is the inverse of the
    covariance matrix

    *L* is the cholesky decomposition of H+D, where D is a small correction
    to force H+D to be positive definite.  To compute parameter uncertainty

    *iterations* is the number of iterations

    *evals* is the number of function evaluations

    *linesearch_evals* is the number of function evaluations for line search
    """
    # print("starting QN")
    # If some input parameters are not specified, define default values for them
    # here. First and second parameters fn and x0 must be defined, others may be
    # passed.  If you want to set a value to a parameter, say to typf, make
    # sure all the parameters before this parameter are specified, in this
    # case fn, x0, grad, and Sx if you want to have default values for grad
    # and Sx, for each enter [].
    # important for also computing fcount (function count)
    n = len(x0)
    if x0 is None:
        x0 = zeros(n)

    if grad is None:
        analgrad = 0
    else:
        analgrad = 1

    if Sx is None:
        Sx = ones(n)
        #Sx = x0 + (x0==0.)
    elif len(Sx) != n:
        raise ValueError("sizes of x0 and Sx must be the same")

    if macheps is None:
        # PAK: use finfo rather than macheps
        macheps = finfo('d').eps

    if eta is None:
        eta = macheps

    fcount = 0                    # total function count
    fcount_ls = 0                # funciton count due to line search

    # If analytic gradient is available then fn will return both function
    # value and analytic gradient.  Otherwise, use finite difference method
    # for estimating the gradient
    if analgrad == 1:
        fc = fn(x0)
        gc = grad(x0)
        fcount = fcount + 1
    else:
        fc = fn(x0)
        gc = fdgrad(n, x0, fc, fn, Sx, eta)
        fcount = fcount + n + 1

    # Check if the initial guess is a local minimizer
    termcode = umstop0(n, x0, fc, gc, Sx, typf, gradtol)
    consecmax = 0

    # Value to return if we fail early
    # Approximately x0 is a critical point
    xf = x0
    ff = fc
    H = L = None
    if termcode == 0:
        H = inithessunfac(n, fc, typf, Sx)

    # STEP 9.
    xc = x0

    # Iterate until convergence in the following loop
    itncount = 0
    while termcode == 0:  # todo. increase itncount
        # print("update",itncount)
        itncount = itncount + 1

        # disp(['Iteration = ' num2str(itncount)])
        # Find Newton step sN
        H, L = modelhess(n, Sx, macheps, H)
        # the vector obtained in the middle
        middle_step_v = linalg.solve(L, -gc)
        sN = linalg.solve(L.transpose(), middle_step_v)   # the last step
        if isnan(sN).any():
            # print("H",H)
            # print("L",L)
            # print("v",middle_step_v)
            # print("Sx",Sx)
            # print("gc",gc)
            termcode = 9
            break

        # Perform line search (Alg.6.3.1). todo. put param order as in the book
        # print("calling linesearch",xc,fc,gc,sN,Sx,H,L,middle_step_v)
        # print("linesearch",xc,fc)
        retcode, xp, fp, maxtaken, fcnt \
            = linesearch(fn, n, xc, fc, gc, sN, Sx, maxstep, steptol)
        fcount += fcnt
        fcount_ls += fcnt
        #plot(xp(1), xp(2), 'g.')

        # Evaluate gradient at new point xp
        if analgrad == 1:
            gp = grad(xp)
        else:
            gp = fdgrad(n, xp, fp, fn, Sx, eta)
            fcount = fcount + n

        # Check stopping criteria (alg.7.2.1)
        consecmax = consecmax + 1 if maxtaken else 0
        termcode = umstop(n, xc, xp, fp, gp, Sx, typf, retcode, gradtol,
                          steptol, itncount, itnlimit, consecmax)

        if abort_test():
            termcode = 6

        # STEP 10.6
        # If termcode is larger than zero, we found a point satisfying one
        # of the termination criteria, return from here.  Otherwise evaluate
        # the next Hessian approximation (Alg. 9.4.1).
        if termcode > 0:
            xf = xp                                        # x final
            ff = fp                                        # f final

        elif not monitor(x=xp, fx=fp, step=itncount):
            termcode = 6

        else:
            H = bfgsunfac(n, xc, xp, gc, gp, macheps, eta, analgrad, H)
            xc = xp
            fc = fp
            gc = gp
        # STOPHERE

    result = dict(status=termcode, x=xf, fx=ff, H=H, L=L,
                  iterations=itncount, evals=fcount, linesearch_evals=fcount_ls)
    #print("result",result, steptol, macheps)
    return result

#------------------------------------------------------------------------------
#@author: Ismet Sahin
# Alg. 9.4.1

# NOTE:
# BFCG Hessian update is performed unless the following two conditions hold
#    (i) y'*s < sqrt(macheps)*norm(s)*norm(y)
#    (ii)


def bfgsunfac(n, xc, xp, gc, gp, macheps, eta, analgrad, H):
    s = xp - xc
    y = gp - gc
    temp1 = inner(y, s)
    # ISMET : I added condition of having temp1 != 0
    if temp1 >= sqrt(macheps) * linalg.norm(s) * linalg.norm(y) and temp1 != 0:
        if analgrad == 1:
            tol = eta
        else:
            tol = sqrt(eta)

        # deal with noise levels in y
        skipupdate = 1
        t = dot(H, s)
        temp_logicals = (abs(y - t) >= tol * maximum(abs(gc), abs(gp)))
        if sum(temp_logicals):
            skipupdate = 0

        # do the BFGS update if skipdate is false
        if skipupdate == 0:
            temp2 = dot(s, t)
            H = H + outer(y, y) / temp1 - outer(t, t) / temp2

    return H


#------------------------------------------------------------------------------
'''
@author: Ismet Sahin
'''


def choldecomp(n, H, maxoffl, macheps):
    minl = (macheps) ** (0.25) * maxoffl

    if maxoffl == 0:
        # H is known to be a positive definite matrix
        maxoffl = sqrt(max(abs(diag(H))))

    minl2 = sqrt(macheps) * maxoffl

    # 3. maxadd is the number (R) specifying the maximum amount added to any
    # diagonal entry of Hessian matrix H
    maxadd = 0

    # 4. form column j of L
    L = zeros((n, n))
    for j in range(1, n + 1):
        L[j - 1, j - 1] = H[j - 1, j - 1] - sum(L[j - 1, 0:j - 1] ** 2)
        minljj = 0
        for i in range(j + 1, n + 1):
            L[i - 1, j - 1] = H[j - 1, i - 1] - \
                sum(L[i - 1, 0:j - 1] * L[j - 1, 0:j - 1])
            minljj = max(abs(L[i - 1, j - 1]), minljj)

        # 4.4
        minljj = max(minljj / maxoffl, minl)

        # 4.5
        if L[j - 1, j - 1] > minljj ** 2:
            # normal Cholesky iteration
            L[j - 1, j - 1] = sqrt(L[j - 1, j - 1])
        else:
            # augment H[j-1,j-1]
            if minljj < minl2:
                minljj = minl2    # occurs only if maxoffl = 0

            maxadd = max(maxadd, minljj ** 2 - L[j - 1, j - 1])
            L[j - 1, j - 1] = minljj

        # 4.6
        L[j:n, j - 1] = L[j:n, j - 1] / L[j - 1, j - 1]

    return L, maxadd

#------------------------------------------------------------------------------
# ALGORITHM 5.6.3

# Ismet Sahin

# function g = fdgrad(n, xc, fc, objfunc, sx, eta)
# g = fdgrad(@obj_function1, 2, [1 -1]', 10, [1 1], eps)

# NOTATION:
#    N : Natural number
#    R : Real number
#    Rn: nx1 real vector
#    Rnxm : nxm real matrix

# INPUTS:
#    n  : the dimension of the gradient vector (N)
#    xc : the current point at which the value of gradient is computed (Rn)
#    fc : function value at xc (R)
#    objfunc : a function handle which is used to compute function values
#    Sx : a n-dim vector, jth entry specifies the typical value of jth param.
# (Rn)
#    eta: equals to 1e-DIGITS where DIGITS is an integer specifying the
# number of reliable digits (R)
# OUTPUT:
#    g : the n-dim finite difference gradient vector (Rn)

# NOTES :
#    hj : is the constant specifying the step size in the direction of jth
# coordinate (R)
#    ej : the unit vector, jth column of the identity matrix (Rn)

# COMMENTS:
#--- FIND STEP SIZE hj
#    1.a : sign(x) does not work for us when x = 0 since this makes the step
# size hj zero which is not allowed. (Step size = 0 => gj = inf.)
#    1.b : evaluation of the step size
#    1.c : a trick to reduce error due to finite precision.  The line xc(j) =
# xc(j) + hj is equivalent to xc = xc + hj * ej where ej is the jth column
# of identity matrix.
#
#--- EVALUATE APPR. GRADIENT
# First evaluate function at xc + hj * ej and then estimate jth entry of
# the gradient.


def fdgrad(n, xc, fc, fn, Sx, eta):

    # create memory for gradient
    g = zeros(n)

    sqrteta = sqrt(eta)
    for j in range(1, n + 1):
        #--- FIND STEP SIZE hj
        if xc[j - 1] >= 0:
            signxcj = 1
        else:
            signxcj = -1                # 1.a

        # 1.b
        hj = sqrteta * max(abs(xc[j - 1]), 1 / Sx[j - 1]) * signxcj

        # 1.c
        tempj = xc[j - 1]
        xc[j - 1] = xc[j - 1] + hj
        hj = xc[j - 1] - tempj

        #--- EVALUATE APPR. GRADIENT
        fj = fn(xc)
        # PAK: hack for infeasible region: point the other way
        if isinf(fj):
            fj = fc + hj
        g[j - 1] = (fj - fc) / hj
        # if isinf(g[j-1]):
        #    print("fc,fj,hj,Sx,xc",fc,fj,hj,Sx[j-1],xc[j-1])

        # now reset the current
        xc[j - 1] = tempj

    #print("gradient", g)
    return g


#------------------------------------------------------------------------------
# @author: Ismet Sahin
# Example call:
# H = inithessunfac(2, f, 1, [1 0.1]')

def inithessunfac(n, f, typf, Sx):
    temp = max(abs(f), typf)
    H = diag(temp * Sx ** 2)
    return H


#------------------------------------------------------------------------------

def linesearch(cost_func, n, xc, fc, g, p, Sx, maxstep, steptol):
    """
ALGORITHM 6.3.1

Ismet Sahin

THE PURPOSE

    is to find a step size which yields the new function value smaller than the
    current function value, i.e. f(xc + alfa*p) <= f(xc) + alfa * lambda * g'p

CONDITIONS

    g'p < 0
    alfa < 0.5

NOTATION:
    N : Natural number
    R : Real number
    Rn: nx1 real vector
    Rnxm : nxm real matrix
    Str: a string

INPUTS
    n : dimensionality (N)
    xc : the current point ( Rn)
    fc : the function value at xc (R)
    obj_func : the function handle to evaluate function values (str like :
       '@costfunction1')
    g : gradient (Rn)
    p : the descent direction (Rn)
    Sx : scale factors (Rn)
    maxstep : maximum step size allowed (R)
    steptol : step tolerance in order to break infinite loop in line search (R)

OUTPUTS
    retcode : boolean indicating a new point xp found (0) or not (1)    (N).
    xp : the new point (Rn)
    fp : function value at xp (R)
    maxtaken : boolean (N)

NOTES:
    alfa : is used to prevent function value reductions which are too small.
       Here we'll use a very small number in order to accept very small
       reductions but not too small.
"""

    maxtaken = 0

    # alfa specifies how much function value reduction is allowable.  The
    # smaller the alfa, the smaller the function value reduction we allow.
    alfa = 1e-4

    # the magnitude of the Newton step
    Newtlen = linalg.norm(Sx * p)

    if Newtlen > maxstep:
        # Newton step is larger than the max acceptable step size (maxstep).
        # Make it equal or smaller than maxstep
        p = p * (maxstep / Newtlen)
        Newtlen = maxstep

    initslope = inner(g, p)

    # "Relative length of p as calculated in the stopping routine"
    # rellength = amax(abs(p) / maximum(abs(xc), Sx))    # this was a bug
    rellength = amax(abs(p) / maximum(abs(xc), 1 / Sx))

    minlambda = steptol / rellength

    lambdaM = 1.0

    # In this loop, we try to find an acceptable next point
    # xp = xc + lambda * p by finding an optimal lambda based on one
    # dimensional quadratic and cubic models
    fcount = 0
    while True:                # 10 starts.
        # next point candidate
        xp = xc + lambdaM * p
        if isnan(xp).any():
            #print("nan xp")
            retcode = 1
            xp, fp = xc, fc
            break
        if fcount > 20:
            #print("too many cycles in linesearch",xp)
            retcode = 2
            xp, fp = xc, fc
            break
        # function value at xp
        fp = cost_func(xp)
        #print("linesearch",fcount,xp,xc,lambdaM,p,fp,fc)
        if isinf(fp):
            fp = 2 * fc  # PAK: infeasible region hack
        fcount = fcount + 1
        if fp <= fc + alfa * lambdaM * initslope:
            # satisfactory xp is found
            retcode = 0
            if lambdaM == 1.0 and Newtlen > 0.99 * maxstep:
                maxtaken = 1
            # return from here
            break
        elif lambdaM < minlambda:
            # step length is too small, so a satisfactory xp cannot be found
            #print("step",lambdaM,minlambda,steptol,rellength)
            retcode = 3
            xp, fp = xc, fc
            break
        else:                            # 10.3c starts
            # reduce lambda by a factor between 0.1 and 0.5
            if lambdaM == 1.0:
                # first backtrack with one dimensional quadratic fit
                lambda_temp = -initslope / (2.0 * (fp - fc - initslope))
                #print("L1",lambda_temp)
            else:
                # perform second and following backtracks with cubic fit
                Mt = array([[1.0/lambdaM**2, -1.0/lambda_prev**2],
                            [-lambda_prev/lambdaM**2, lambdaM/lambda_prev**2]])
                vt = array([[fp - fc - lambdaM * initslope],
                            [fp_prev - fc - lambda_prev * initslope]])
                ab = (1.0 / (lambdaM - lambda_prev)) * dot(Mt, vt)
                # a = ab(1) and b = ab(2)
                disc = ab[1, 0] ** 2 - 3.0 * ab[0, 0] * initslope
                #print("Mt,vt,ab,disc",Mt,vt,ab,disc)
                if ab[0, 0] == 0.0:
                    # cubic model turn out to be a quadratic
                    lambda_temp = -initslope / (2.0 * ab[1, 0])
                    #print("L2",lambda_temp)
                else:
                    # the model is a legitimate cubic
                    lambda_temp = (-ab[1, 0] + sqrt(disc)) / (3.0 * ab[0, 0])
                    #print("L3",lambda_temp)

                if lambda_temp > 0.5 * lambdaM:
                    # larger than half of previous lambda is not allowed.
                    lambda_temp = 0.5 * lambdaM
                    #print("L4",lambda_temp)

            lambda_prev = lambdaM
            fp_prev = fp
            if lambda_temp <= 0.1 * lambdaM:
                # smaller than 1/10 th of previous lambda is not allowed.
                lambdaM = 0.1 * lambdaM
            else:
                lambdaM = lambda_temp

            #print('lambda = ', lambdaM)

    # return xp, fp, retcode
    return retcode, xp, fp, maxtaken, fcount


#------------------------------------------------------------------------------

# @author: Ismet Sahin
# ALGORITHM 1.3.1
def machineeps():
    macheps = 1.0
    while (macheps + 1) != 1:
        macheps = macheps / 2

    macheps = 2 * macheps
    return macheps


#------------------------------------------------------------------------------

def modelhess(n, Sx, macheps, H):
    """
@author: Ismet Sahin.
Thanks to Christopher Meeting for his help in converting this module from
Matlab to Python

ALGORITHM 5.5.1

NOTES:
    Currently we are not implementing steps 1, 14, and 15 (TODO)

This function performs perturbed Cholesky decomposition (CD) as if the input
Hessian matrix is positive definite.  The code for perturbed CD resides in
choldecomp.m file which returns the factored lower triangle matrix L and a
number, maxadd, specifying the largest number added to a diagonal element of
H during the CD decomposition.  This function checks if the decomposition is
completed without adding any positive number to the diagonal elements of H,
i.e. maxadd <= 0.  Otherwise, this function adds the least number to the
diagonals of H which makes it positive definite based on maxadd and other
entries in H.
EXAMPLE CALLS::

         A1 =[2     0    2.4
              0     2     0
              2.4     0     3]

         A2 =[2     0    2.5
               0     2     0
              2.5     0     3]

         A3 =[2     0    10
               0     2     0
              10     0     3]
"""

    # SCALING
    scale_needed = 0                        # ISMET uses this parameter
    if sum(Sx - ones(n)) != 0:
        # scaling is requested by the user
        scale_needed = 1
        Dx = diag(Sx)
        Dx_inv = diag(1.0 / Sx)
        H = dot(Dx_inv, dot(H, Dx_inv))

    # STEP I.
    sqrteps = sqrt(macheps)

    # 2-4.
    H_diag = diag(H)
    maxdiag = max(H_diag)
    mindiag = min(H_diag)

    # 5.
    maxposdiag = max(0, maxdiag)

    # 6. mu is the amount to be added to diagonal of H before the
    # Cholesky decomp. If the minimum diagonal is much much smaller than
    # the maximum diagonal element then adjust mu accordingly otherwise mu = 0.
    if mindiag <= sqrteps * maxposdiag:
        mu = 2 * (maxposdiag - mindiag) * sqrteps - mindiag
        maxdiag = maxdiag + mu
    else:
        mu = 0

    # 7. maximum of off-diagonal elements of H
    diag_infinite = diag(inf * ones(n))
    maxoff = (H - diag_infinite).max()

    # 8. if maximum off diagonal element is much much larger than the maximum
    # diagonal element of the Hessian H
    if maxoff * (1 + 2 * sqrteps) > maxdiag:
        mu = mu + (maxoff - maxdiag) + 2 * sqrteps * maxoff
        maxdiag = maxoff * (1 + 2 * sqrteps)

    # 9.
    if maxdiag == 0:            # if H == 0
        mu = 1
        maxdiag = 1

    # 10. mu>0 => need to add mu amount to the diagonal elements: H = H + mu*I
    if mu > 0:
        diag_mu = diag(mu * ones(n))
        H = H + diag_mu

    # 11.
    maxoffl = sqrt(max(maxdiag, maxoff / n))

    # STEP II. Perform perturbed Cholesky decomposition H + D = LL' where D is
    # a diagonal matrix which is implicitly added to H if H is not positive
    # definite. Matrix D has only positive elements. The output variable maxadd
    # indicates the maximum number added to a diagonal entry of the Hesian,
    # i.e. the maximum of D. If maxadd is returned 0, then H was indeed pd
    # and L is the resulting factor.
    # 12.
    L, maxadd = choldecomp(n, H, maxoffl, macheps)

    # STEP III.
    # 13. If maxadd <= 0, we are done H was positive definite.
    if maxadd > 0:
        # H was not positive definite
        # print('WARNING: Hessian is not pd. Max number added to H is ',maxadd)
        maxev = H[0, 0]
        minev = H[0, 0]
        for i in range(1, n + 1):
            offrow = sum(abs(H[0:i - 1, i - 1])) + sum(abs(H[i - 1, i:n]))
            maxev = max(maxev, H[i - 1, i - 1] + offrow)
            minev = min(minev, H[i - 1, i - 1] - offrow)

        sdd = (maxev - minev) * sqrteps - minev
        sdd = max(sdd, 0)
        mu = min(maxadd, sdd)
        H = H + diag(mu * ones(n))
        L, maxadd = choldecomp(n, H, 0, macheps)

    if scale_needed:                # todo. this calculation can be done faster
        H = dot(Dx, dot(H, Dx))
        L = dot(Dx, L)

    return H, L


#------------------------------------------------------------------------------
def umstop(n, xc, xp, f, g, Sx, typf, retcode, gradtol, steptol,
           itncount, itnlimit, consecmax):
    """
#@author: Ismet Sahin

ALGORITHM 7.2.1

Return codes:
Note that return codes are nonnegative integers. When it is not zero, there is
a termination condition which is satisfied.
   0 : None of the termination conditions is satisfied
   1 : Magnitude of scaled grad is less than gradtol; this is the primary
       condition. The new point xp is most likely a local minimizer.  If gradtol
       is too large, then this condition can be satisfied easier and therefore
       xp may not be a local minimizer
   2 : Scaled distance between last two points is less than steptol; xp might be
       a local minimizer.  This condition may also be satisfied if step is
       chosen too large or the algorithm is far from the minimizer and making
       small progress
   3 : The algorithm cannot find a new point giving smaller function value than
       the current point.  The current may be a local minimizer, or analytic
       gradient implementation has some mistakes, or finite difference gradient
       estimation is not accurate, or steptol is too large.
   4 : Maximum number of iterations are completed
   5 : The maximum step length maxstep is taken for last ten consecutive
       iterations.  This may happen if the function is not bounded from below,
       or the function has a finite asymptote in some direction, or maxstep is
       too small.
    """

    termcode = 0
    if retcode == 1:
        termcode = 3
    elif retcode == 2:
        termcode = 7
    elif retcode == 3:
        termcode = 8
    elif retcode > 0:
        raise ValueError("Unknown linesearch return code")
    elif max(abs(g) * maximum(abs(xp), 1 / Sx) / max(abs(f), typf)) <= gradtol:
        # maximum component of scaled gradient is smaller than gradtol.
        # TODO: make sure not to use a too large typf value which leads to the
        # satisfaction of this algorithm easily.
        termcode = 1
    elif max(abs(xp - xc) / maximum(abs(xp), 1 / Sx)) <= steptol:
        # maximum component of scaled step is smaller than steptol
        termcode = 2
    elif itncount >= itnlimit:
        # maximum number of iterations are performed
        termcode = 4
    elif consecmax == 10:
        # not more than 10 steps will be taken consecutively.
        termcode = 5

    return termcode


#------------------------------------------------------------------------------
#@author: Ismet Sahin

# This function checks whether initial conditions are acceptable for
# continuing unconstrained optimization

# f : the function value at x0, i.e. f = f(x0),  (R)
# g : the gradient at x0, (Rn)

# termcode = 0 : x0 is not a critical point of f(x), (Z)
# termcode = 1 : x0 is a critical point of f(x), (Z)

# Note that x0 may be a critical point of the function; in this case, it is
# either a local minimizer or a saddle point of the function.  If the Hessian
# at x0 is positive definite than it is indeed a local minimizer.  Instead of
# checking Hessian, we can also restart the driver program umexample from
# another point which is close to x0.  If x0 is the local minimizer, the
# algorithm will approach it.

def umstop0(n, x0, f, g, Sx, typf, gradtol):
    #consecmax = 0
    if max(abs(g) * maximum(abs(x0), 1./Sx)/max(abs(f), typf)) <= 1e-3*gradtol:
        termcode = 1
    else:
        termcode = 0
    return termcode


#------------------------------------------------------------------------------

def example_call():
    print('***********************************')

    # Rosenbrock function
    fn = lambda p: (1 - p[0])**2 + 100*(p[1] - p[0]**2)**2
    grad = lambda p: array([-2*(1 - p[0]) - 400*(p[1] - p[0]**2)*p[0],
                            200*(p[1] - p[0]**2)])
    x0 = array([2.320894, -0.534223])
    # x0 = array([2.0,1.0])

    result = quasinewton(fn=fn, x0=x0, grad=grad)
    #result = quasinewton(fn=fn, x0=x0)

    print('\n\nInitial point x0 = ', x0, ', f(x0) = ', fn(x0))
    for k in sorted(result.keys()):
        print(k, "=", result[k])


if __name__ == "__main__":
    example_call()
