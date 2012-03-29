'''
% Ismet Sahin
% Oct 25, 2009

% THE PURPOSE
%    is to find a step size which yields the new function value smaller than the
%    current function value, i.e. f(xc + alfa*p) <= f(xc) + alfa * lambda * g'p

% CONDITIONS
%    g'p < 0
%    alfa < 0.5

% NOTATION:
%    N : Natural number
%    R : Real number
%    Rn: nx1 real vector
%    Rnxm : nxm real matrix
%    Str: a string

% INPUTS
%    n : dimensionality (N)
%    xc : the current point ( Rn)
%    fc : the function value at xc (R)
%    obj_func : the function handle to evaluate function values (str like :
%       '@costfunction1')
%    g : gradient (Rn)
%    p : the descent direction (Rn)
%    Sx : scale factors (Rn)
%    maxstep : maximum step size allowed (R)
%    steptol : step tolerance in order to break infinite loop in line search (R)

% OUTPUTS
%    retcode : boolean indicating a new point xp found (0) or not (1)    (N).
%    xp : the new point (Rn)
%    fp : function value at xp (R)
%    #maxtaken : boolean (N)

% NOTES:
%    alfa : is used to prevent function value reductions which are too small.
%       Here we'll use a very small number in order to accept very small
%       reductions but not too small.
'''

from numpy import linalg, array, dot, sqrt, amax, maximum

def linesearch(cost_func, n, xc, fc, g, p, Sx, maxstep, steptol):
    #maxtaken = 0.0
    retcode = 1.0

    # alfa specifies how much function value reduction is allowable.  The smaller
    # the alfa, the smaller the function value reduction we allow.
    alfa = 1e-4

    # the magnitude of the Newton step
    Newtlen = linalg.norm(Sx * p)

    # TODO : understand how to choose maxstep and steptol.
    if Newtlen > maxstep:
        # Newton step is larger than the maximum acceptable step size (maxstep). Make
        # it equal or smaller than maxstep
        p = p * (maxstep / Newtlen)
        Newtlen = maxstep

    initslope = dot(g.T,p)[0,0]

    # "Relative length of p as calculated in the stopping routine"
    rellength = amax(abs(p) / maximum(abs(xc), Sx))

    minlambda = steptol / rellength

    lambdaM = 1.0

    # In this loop, we try to find an acceptable next point xp = xc + lambda * p by
    # finding an optimal lambda based on one dimensional quadratic and cubic models
    while retcode < 2.0:                # 10 starts.
        xp = xc + lambdaM * p                                    # next point candidate
        fp,gp = cost_func(xp)                                    # function value at xp
        if fp <= fc + alfa * lambdaM * initslope:
            # satisfactory xp is found
            retcode = 0.0
            #if lambdaM == 1.0 and Newtlen > 0.99 * maxstep:
            #    maxtaken = 1.0
            break                                                    # return from here
        elif lambdaM < minlambda:
            # step length is too small, therefore a satisfactory xp cannot be found
            retcode = 1.0
            xp = xc
            break
        else:                            # 10.3c starts
            # reduce lambda by a factor between 0.1 and 0.5
            if lambdaM == 1.0:
                # first backtrack with one dimensional quadratic fit
                lambda_temp = -initslope / (2.0*(fp-fc-initslope))
            else:
                # ISMET : I added the following if statements
                if lambdaM == lambda_prev:
                    print 'Warning : lambda is equal to the previous lambda'
                    break

                # perform second and following backtracks with cubic fit
                Mt = array([[1.0/(lambdaM**2), -1.0/(lambda_prev**2)],
                            [-lambda_prev/(lambdaM**2), lambdaM/(lambda_prev**2)]])
                vt = array([[fp - fc - lambdaM * initslope],
                            [fp_prev - fc - lambda_prev * initslope]])
                ab = (1.0/(lambdaM-lambda_prev)) * dot(Mt,vt)
                disc = ab[1,0]**2 - 3.0 * ab[0,0] * initslope        # a = ab(1) and b = ab(2)
                if ab[0,0] == 0.0:
                    # cubic model turn out to be a quadratic
                    lambda_temp = -initslope / (2.0*ab[1,0])
                else:
                    # the model is a legitimate cubic
                    lambda_temp = (-ab[1,0] + sqrt(disc)) / (3.0 * ab[0,0])

                if lambda_temp > 0.5 * lambdaM:
                    # larger than half of previous lambda is not allowed.
                    lambda_temp = 0.5 * lambdaM

            lambda_prev = lambdaM
            fp_prev = fp

            if lambda_temp <= 0.1 * lambdaM:
                # smaller than 1/10 th of previous lambda is not allowed.
                lambdaM = 0.1 * lambdaM
            else:
                lambdaM = lambda_temp

    return xp, fp, retcode
