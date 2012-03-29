'''
% Ismet Sahin
% Sep 29, 2009

% INPUTS:
%    obj_function : a function handle returning function value and gradient
%                        value at a point x
%    x : initial guess
%    max_iter : maximum number of iterations allowed
%    epsilon : stop iterations when the norm of gradient becomes smaller than
%                 epsilon,
% OUTPUTS:
%    x_star = optimum found by the algorithm

% NOTES:
#    We assume that (i) the Hessian matrix is invertable and (ii) it is
#    positive definite (pd).  Both of these assumption may easily fail for some
#    optimization problems and they need to be addressed. In literature ILU
#    (incomplete LU) factorization seems to be very popular for doing the
#    Newton step; check this.  When Hessian is not pd, consider adding a
#    positive real number to the diagonal entries.
#
#    We have chosen the step size alfa = 1 which needs to be addressed as
#    well.  There are many choice for step size.  A popular one is to
#    quadratic or cubic line search.
#
#    Currently we assumed that the analytic form of the gradient is
#    available.  This can be estimated from only the function evaluations
#    based on forward or central gradient estimation methods.
#
#    There are multiple algorithms for constructing the initial Hessian
#    matrix.  Implement a few of them.  Currently, we make sure that the
#    initial Hesian is the scaled identity matrix whose diagonals are large
#    enough for preventing very large initial Newton steps.
'''

from numpy import sqrt
from numpy import (eye, zeros, ones, dot, outer, array, linalg, finfo, asarray)
eps = finfo(type(1.)).eps
import modelhess as mh
import linesearch as ls

def bfgs_cholesky_backtracking(fn, x0, max_iter=500, vtr=1e-6, verbose=False):
    # Wrap cost function fn so that it takes a row vector and return
    # value plus column vector.
    n = len(x0)
    x = asarray(x0)[:,None]
    def cost_func(x):
        cost_func.calls += 1
        f,g = fn(x[:,0])
        return f,asarray(g)[:,None]
    cost_func.calls = 0

    # ----- Initial step -----
    # initial Hessian matrix H is the Identity matrix
    f, g = cost_func(x)         # function value and gradient
    scale = max(f, 1)           # scale diagonals of init Hessian
    H = scale * eye(n)          # initial Hessian

    #x_star = Inf

    fv = array(zeros((max_iter, 1)))

    for k in range(1, max_iter+1):
        if verbose: print k, x[:,0]

        # Newton step
        L, H = mh.modelhess(n, H, 1, eps)
        y = linalg.solve(L, -g)
        p = linalg.solve(L.T, y)

        Sx = ones((n,1))
        maxstep = 10
        steptol = 1e-6
        x_new, f_new, retcode = ls.linesearch(cost_func, n, x, f, g, p, Sx, maxstep, steptol)

        f_new,g_new = cost_func(x_new)

        # Gradient difference vector
        y = g_new - g

        # Hessian estimate  TODO :
        if dot(y.T,p) > sqrt(eps) * linalg.norm(y) * linalg.norm(p): # ISMET : I added this if statement based on cond in Algorithm A941
            p_rot = dot(H,p)
            H = H + outer(y, y) / dot(y.T, p) - outer(p_rot, p_rot) / dot(p.T, p_rot)

        # Update parameters
        g = g_new
        x = x_new
        f = f_new                                    # currently not used

        fv[k-1] = f

        # Check for convergence
        if linalg.norm(g) < vtr:
            print 'Convergence is achieved after', k, 'iterations', cost_func.calls,'calls'
            #x_star = x_new
            return x[:,0], fv
    else:
        print 'Failed to converge after',k,'iterations',cost_func.calls,'calls'
        return None,None



def check(scale=10):
    import scipy.optimize as opt
    import numpy
    from cost_func import rosen
    fail = 0
    sfail = 0
    for i in range(100):
        x0 = numpy.random.uniform(size=(4,))
        print "***Trying",i,x0
        x,fv = bfgs_cholesky_backtracking(rosen,x0)
        if x is None: fail += 1
        scipy_x,scipy_fv,g,H,fun,grad,warn = opt.fmin_bfgs(opt.rosen,x0,opt.rosen_der,full_output=True)
        if warn>0: sfail += 1

    print "sahin failed",fail,"out of",100
    print "scipy failed",sfail,"out of",100

def main(name='rosen'):
    if name is 'rosen':
        from cost_func import rosen
        f,x0 = rosen, [20,10,10,20]
    else:
        f = lambda x: ((x[0]+x[1]-1)**2,[2*(x[0]+x[1]),2*(x[0]+x[1])])
        x0 = 5,10
    x,fv = bfgs_cholesky_backtracking(f, x0, verbose=True)
    print 'minimizer is : \n', x
    import pylab
    pylab.semilogy(fv)
    pylab.show()

if __name__ == "__main__":
    #main('rosen')
    #main('ab')
    check(3)
