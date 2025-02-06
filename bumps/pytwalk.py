"""
T-walk self adjusting MCMC.
"""

# Author: By Andres Christen.
# see:  http://www.cimat.mx/~jac/twalk/

# 2010-04-17 Paul Kienzle
# * typo fixups
# * move pylab import to the particular functions
# * remove scipy dependence

__all__ = ["pytwalk"]

from numpy.random import uniform, normal
from numpy import ones, zeros, cumsum, shape, asmatrix as mat, cov, mean, ceil, matrix, sqrt
from numpy import floor, exp, log, sum, pi, arange

# Some auxiliary functions and constants
# square of the norm.


def SqrNorm(x):
    return sum(x * x)


log2pi = log(2 * pi)


class pytwalk:
    """This is the t-walk class.

    Initiates defining the dimension= n and -log of the objective function= U,
    Supp defines the support, returns True if x within the support, eg:

    Mytwalk = twalk( n=3, U=MyMinusLogf, Supp=MySupportFunction).

    Then do: Mytwalk.Run?

    Other parameter are:
    ww= the prob. of choosing each kernel, aw, at, n1phi (see inside twalk.py)
    with default values as in the paper, normally NOT needed to be changed."""

    def __init__(
        self,
        n,
        U=(lambda x: sum(0.5 * x**2)),
        Supp=(lambda x: True),
        ww=(0.0000, 0.4918, 0.4918, 0.0082, 0.0082),
        aw=1.5,
        at=6.0,
        n1phi=4.0,
    ):
        self.n = n
        self.U = U
        self.Supp = Supp
        self.Output = zeros((1, n + 1))  # No data (MCMC output) yet
        self.T = 1
        # To save the acceptance rates of each kernel, and the global acc. rate
        self.Acc = zeros(6)

        # Kernel probabilities
        self.Fw = cumsum(ww)

        # Parameters for the propolsals
        self.aw = aw  # For the walk move
        self.at = at  # For the Traverse move

        # n1phi = 5 ### expected value of parameters to move
        self.pphi = min(n, n1phi) / (1.0 * n)  # Prob. of choosing each par.

    def Run(self, T, x0, xp0):
        """Run the twalk.

        Run( T, x0, xp0),
        T = Number of iterations.
        x0, xp0, two initial points within the support,
        ***each entry of x0 and xp0 most be different***.
        """

        print("twalk: Running the twalk with %d iterations." % T)
        # Check x0 and xp0 in the support
        x = x0  # Reference, so we can retrieve the last values used
        if not (self.Supp(x)):
            print("twalk: ERROR, initial point x0 = %12.4g out of support." % x0)
            return 0
        u = self.U(x)

        xp = xp0
        if not (self.Supp(xp)):
            print("twalk: ERROR, initial point xp0 = %12.4g out of support." % xp0)
            return 0
        up = self.U(xp)

        if any(abs(x0 - xp0) <= 0):
            print("twalk: ERROR, not all entries of initial values different.")
            return 0

        # Set the array to place the iterations and the U's ... we donot save
        # up's
        self.Output = zeros((T + 1, self.n + 1))
        self.T = T + 1
        self.Acc = zeros(6)
        kercall = zeros(6)  # Times each kernel is called

        # Make local references for less writing
        n = self.n
        Output = self.Output
        Acc = self.Acc

        Output[0, 0:n] = x.copy()
        Output[0, n] = u

        # Sampling
        for it in range(T):
            y, yp, ke, A, u_prop, up_prop = self.onemove(x, u, xp, up)

            kercall[ke] += 1
            kercall[5] += 1
            if uniform() < A:
                x = y.copy()  # Accept the proposal y
                u = u_prop
                xp = yp.copy()  # Accept the proposal yp
                up = up_prop

                Acc[ke] += 1
                Acc[5] += 1

            # To retrive the current values
            self.x = x
            self.xp = xp
            self.u = u
            self.up = up

            Output[it + 1, 0:n] = x.copy()
            Output[it + 1, n] = u

        if Acc[5] == 0:
            print("twalk: WARNING,  all propolsals were rejected!")
            return 0

        for i in range(6):
            if kercall[i] != 0:
                Acc[i] /= kercall[i]
        return 1

    def onemove(self, x, u, xp, up):
        """One move of the twalk.  This is basically the raw twalk kernel.
        It is usefull if the twalk is needed inside a more complex MCMC.

        onemove(x, u, xp, up),
        x, xp, two points WITHIN the support ***each entry of x0 and xp0 must be different***.
        and the value of the objective at x, and xp
        u=U(x), up=U(xp).

        It returns: [y, yp, ke, A, u_prop, up_prop]
        y, yp: the proposed jump
        ke: The kernel used, 0=nothing, 1=Walk, 2=Traverse, 3=Blow, 4=Hop
        A: the M-H ratio
        u_prop, up_prop: The values for the objective func. at the proposed jumps
        """

        # Make local references for less writing
        U = self.U
        Supp = self.Supp
        Fw = self.Fw

        ker = uniform()  # To choose the kernel to be used
        ke = 1
        A = 0

        # Kernel nothing exchange x with xp
        if (0.0 <= ker) & (ker < Fw[0]):
            ke = 0
            y = xp.copy()
            up_prop = u
            yp = x.copy()
            u_prop = up
            # A is the MH acceptance ratio
            A = 1.0
            # always accepted

        # The Walk move
        if (Fw[0] <= ker) & (ker < Fw[1]):
            ke = 1

            dir = uniform()

            if (0 <= dir) & (dir < 0.5):  # x as pivot
                yp = self.SimWalk(xp, x)

                y = x.copy()
                u_prop = u

                if (Supp(yp)) & (all(abs(yp - y) > 0)):
                    up_prop = U(yp)
                    A = exp(up - up_prop)
                else:
                    up_prop = None
                    A = 0
                    # out of support, not accepted

            else:  # xp as pivot
                y = self.SimWalk(x, xp)

                yp = xp.copy()
                up_prop = up

                if (Supp(y)) & (all(abs(yp - y) > 0)):
                    u_prop = U(y)
                    A = exp(u - u_prop)
                else:
                    u_prop = None
                    A = 0
                    # out of support, not accepted

        # The Traverse move
        if (Fw[1] <= ker) & (ker < Fw[2]):
            ke = 2
            dir = uniform()

            if (0 <= dir) & (dir < 0.5):  # x as pivot
                beta = self.Simbeta()
                yp = self.SimTraverse(xp, x, beta)

                y = x.copy()
                u_prop = u

                if Supp(yp):
                    up_prop = U(yp)
                    if self.nphi == 0:
                        A = 1  # Nothing moved
                    else:
                        A = exp((up - up_prop) + (self.nphi - 2) * log(beta))
                else:
                    up_prop = None
                    A = 0  # out of support, not accepted
            else:  # xp as pivot
                beta = self.Simbeta()
                y = self.SimTraverse(x, xp, beta)

                yp = xp.copy()
                up_prop = up

                if Supp(y):
                    u_prop = U(y)
                    if self.nphi == 0:
                        A = 1  # Nothing moved
                    else:
                        A = exp((u - u_prop) + (self.nphi - 2) * log(beta))
                else:
                    u_prop = None
                    A = 0  # out of support, not accepted

        # The Blow move
        if (Fw[2] <= ker) & (ker < Fw[3]):
            ke = 3
            dir = uniform()

            if (0 <= dir) & (dir < 0.5):  # x as pivot
                yp = self.SimBlow(xp, x)

                y = x.copy()
                u_prop = u
                if (Supp(yp)) & all(yp != x):
                    up_prop = U(yp)
                    W1 = self.GBlowU(yp, xp, x)
                    W2 = self.GBlowU(xp, yp, x)
                    A = exp((up - up_prop) + (W1 - W2))
                else:
                    up_prop = None
                    A = 0  # out of support, not accepted
            else:  # xp as pivot
                y = self.SimBlow(x, xp)

                yp = xp.copy()
                up_prop = up
                if (Supp(y)) & all(y != xp):
                    u_prop = U(y)
                    W1 = self.GBlowU(y, x, xp)
                    W2 = self.GBlowU(x, y, xp)
                    A = exp((u - u_prop) + (W1 - W2))
                else:
                    u_prop = None
                    A = 0  # out of support, not accepted

        # The Hop move
        if (Fw[3] <= ker) & (ker < Fw[4]):
            ke = 4
            dir = uniform()

            if (0 <= dir) & (dir < 0.5):  # x as pivot
                yp = self.SimHop(xp, x)

                y = x.copy()
                u_prop = u
                if (Supp(yp)) & all(yp != x):
                    up_prop = U(yp)
                    W1 = self.GHopU(yp, xp, x)
                    W2 = self.GHopU(xp, yp, x)
                    A = exp((up - up_prop) + (W1 - W2))
                else:
                    up_prop = None
                    A = 0  # out of support, not accepted
            else:  # xp as pivot
                y = self.SimHop(x, xp)

                yp = xp.copy()
                up_prop = up
                if (Supp(y)) & all(y != xp):
                    u_prop = U(y)
                    W1 = self.GHopU(y, x, xp)
                    W2 = self.GHopU(x, y, xp)
                    A = exp((u - u_prop) + (W1 - W2))
                else:
                    u_prop = None
                    A = 0  # out of support, not accepted

        return [y, yp, ke, A, u_prop, up_prop]

    ##########################################################################
    # Auxiliaries for the kernels

    # Used by the Walk kernel
    def SimWalk(self, x, xp):
        aw = self.aw
        n = self.n

        phi = uniform(size=n) < self.pphi  # parameters to move
        self.nphi = sum(phi)
        z = zeros(n)

        for i in range(n):
            if phi[i]:
                u = uniform()
                z[i] = (aw / (1 + aw)) * (aw * u**2.0 + 2.0 * u - 1.0)

        return x + (x - xp) * z

    # Used by the Traverse kernel
    def Simbeta(self):
        at = self.at
        if uniform() < (at - 1.0) / (2.0 * at):
            return exp(1.0 / (at + 1.0) * log(uniform()))
        else:
            return exp(1.0 / (1.0 - at) * log(uniform()))

    def SimTraverse(self, x, xp, beta):
        n = self.n

        phi = uniform(size=n) < self.pphi
        self.nphi = sum(phi)

        rt = x.copy()
        for i in range(n):
            if phi[i]:
                rt[i] = xp[i] + beta * (xp[i] - x[i])

        return rt

    # Used by the Blow kernel
    def SimBlow(self, x, xp):
        n = self.n

        phi = uniform(size=n) < self.pphi
        self.nphi = sum(phi)

        self.sigma = max(phi * abs(xp - x))

        rt = x.copy()
        for i in range(n):
            if phi[i]:
                rt[i] = x[i] + self.sigma * normal()

        return rt

    def GBlowU(self, h, x, xp):
        nphi = self.nphi

        if nphi > 0:
            return (nphi / 2.0) * log2pi + nphi * log(self.sigma) + 0.5 * SqrNorm(h - xp) / (self.sigma**2)
        else:
            return 0

    # Used by the Hop kernel
    def SimHop(self, x, xp):
        n = self.n

        phi = uniform(size=n) < self.pphi
        self.nphi = sum(phi)

        self.sigma = max(phi * abs(xp - x)) / 3.0

        rt = x.copy()
        for i in range(n):
            if phi[i]:
                rt[i] = xp[i] + self.sigma * normal()

        return rt

    def GHopU(self, h, x, xp):  # It is actually equal to GBlowU!
        nphi = self.nphi

        if nphi > 0:
            return (nphi / 2.0) * log2pi + nphi * log(self.sigma) + 0.5 * SqrNorm(h - xp) / (self.sigma**2)
        else:
            return 0

    ##########################################################################
    # Output analysis auxiliary methods

    def IAT(self, par=-1, start=0, end=0, maxlag=0):
        """Calculate the Integrated Autocorrelation Times of parameters par
        the default value par=-1 is for the IAT of the U's"""
        if end == 0:
            end = self.T

        if self.Acc[5] == 0:
            print("twalk: IAT: WARNING,  all proposals were rejected!")
            print("twalk: IAT: Cannot calculate IAT, fixing it to the sample size.")
            return self.T

        iat = IAT(self.Output, cols=par, maxlag=maxlag, start=start, end=end)

        return iat

    def TS(self, par=-1, start=0, end=0):
        """Plot time series of parameter par (default = log f) etc."""
        from pylab import plot, xlabel, ylabel

        if par == -1:
            par = self.n

        if end == 0:
            end = self.T

        if par == self.n:
            plot(arange(start, end), -1 * self.Output[start:end, par])
            ylabel("Log of Objective")
        else:
            plot(arange(start, end), self.Output[start:end, par])
            ylabel("Parameter %d" % par)
        xlabel("Iteration")

    def Ana(self, par=-1, start=0, end=0):
        """Output Analysis, TS plots, acceptance rates, IAT etc."""
        if par == -1:
            par = self.n

        if end == 0:
            end = self.T

        print("Acceptance rates for the Walk, Traverse, Blow and Hop kernels:" + str(self.Acc[1:5]))
        print("Global acceptance rate: %7.5f" % self.Acc[5])

        iat = self.IAT(par=par, start=start, end=end)
        print("Integrated Autocorrelation Time: %7.1f, IAT/n: %7.1f" % (iat, iat / self.n))

        self.TS(par=par, start=start, end=end)

    def Hist(self, par=-1, start=0, end=0, g=(lambda x: x[0]), xlab="g", bins=20):
        """Basic histograms and output analysis.  If par=-1, use g.
        The function g provides a transformation to be applied to the data,
        eg g=(lambda x: abs(x[0]-x[1]) would plot a histogram of the distance
        between parameters 0 and 1, etc."""
        from pylab import hist, xlabel

        if end == 0:
            end = self.T

        if par == -1:
            ser = zeros(end - start)
            for it in range(end - start):
                ser[it] = g(self.Output[it + start, :])
            xlabel(xlab)
            print("Mean for %s= %f" % (xlab, mean(ser)))
        else:
            ser = self.Output[start:end, par]
            xlabel("Parameter %d" % par)
            print("Mean for par %d= %f" % (par, mean(ser)))

        hist(ser, bins=bins)
        print("Do:\nfrom pylab import show\nshow()")

    def Save(self, fnam, start=0, thin=1):
        """Saves the Output as a text file, starting at start (burn in), with thinning (thin)."""
        print(("Saving output, all pars. plus the U's in file", fnam))

        from numpy import savetxt

        savetxt(fnam, self.Output[start:,])

    # A simple Random Walk M-H
    def RunRWMH(self, T, x0, sigma):
        """Run a simple Random Walk M-H"""

        print("twalk: This is the Random Walk M-H running with %d iterations." % T)
        # Local variables
        x = x0.copy()
        if not (self.Supp(x)):
            print("twalk: ERROR, initial point x0 out of support.")
            return 0

        u = self.U(x)
        n = self.n

        # Set the array to place the iterations and the U's
        self.Output = zeros((T + 1, n + 1))
        self.Acc = zeros(6)

        # Make local references for less writing
        Output = self.Output
        U = self.U
        Supp = self.Supp
        Acc = self.Acc

        Output[0, 0:n] = x.copy()
        Output[0, n] = u

        y = x.copy()
        for it in range(T):
            y = x + normal(size=n) * sigma  # each entry with sigma[i] variance
            if Supp(y):  # If it is within the support of the objective
                uprop = U(y)  # Evaluate the objective
                if uniform() < exp(u - uprop):
                    x = y.copy()  # Accept the propolsal y
                    u = uprop
                    Acc[5] += 1

            Output[it + 1, 0:n] = x
            Output[it + 1, n] = u

        if Acc[5] == 0:
            print("twalk: WARNING,  all propolsals were rejected!")
            return 0

        Acc[5] /= T
        return 1


##########################################################################
# Auxiliary functions to calculate Integrated autocorrelation times of a
# time series ####


# Calculates an autocovariance 2x2 matrix at lag l in column c of matrix Ser with T rows
# The variances of each series are in the diagonal and the
# (auto)covariance in the off diag.
def AutoCov(Ser, c, la, T=0):
    if T == 0:
        T = shape(Ser)[0]  # Number of rows in the matrix (sample size)

    return cov(Ser[0 : (T - 1 - la), c], Ser[la : (T - 1), c], bias=1)


# Calculates the autocorrelation from lag 0 to lag la of columns cols (list)
# for matrix Ser
def AutoCorr(Ser, cols=0, la=1):
    T = shape(Ser)[0]  # Number of rows in the matrix (sample size)

    ncols = shape(mat(cols))[1]  # Number of columns to analyse (parameters)

    # if ncols == 1:
    #    cols = [cols]

    # Matrix to hold output
    Out = matrix(ones((la) * ncols)).reshape(la, ncols)

    for c in range(ncols):
        for l in range(1, la + 1):
            Co = AutoCov(Ser, cols[c], l, T)
            Out[l - 1, c] = Co[0, 1] / (sqrt(Co[0, 0] * Co[1, 1]))

    return Out


# Makes an upper band matrix of ones, to add the autocorrelation matrix
# gamma = auto[2*m+1,c]+auto[2*m+2,c] etc.
# MakeSumMat(lag) * AutoCorr( Ser, cols=c, la=lag) to make the gamma matrix
def MakeSumMat(lag):
    rows = (lag) / 2  # Integer division!
    Out = mat(zeros([rows, lag], dtype=int))

    for i in range(rows):
        Out[i, 2 * i] = 1
        Out[i, 2 * i + 1] = 1

    return Out


# Finds the cutting time, when the gammas become negative
def Cutts(Gamma):
    cols = shape(Gamma)[1]
    rows = shape(Gamma)[0]
    Out = mat(zeros([1, cols], dtype=int))
    Stop = mat(zeros([1, cols], dtype=bool))

    if rows == 1:
        return Out

    i = 0
    # while (not(all(Stop)) & (i < (rows-1))):
    for i in range(rows - 1):
        for j in range(cols):  # while Gamma stays positive and decreasing
            if ((Gamma[i + 1, j] > 0.0) & (Gamma[i + 1, j] < Gamma[i, j])) & (not Stop[0, j]):
                Out[0, j] = i + 1  # the cutting time for colomn j is i+i
            else:
                Stop[0, j] = True
        i += 1

    return Out


# Automatically find a maxlag for IAT calculations
def AutoMaxlag(Ser, c, rholimit=0.05, maxmaxlag=20000):
    Co = AutoCov(Ser, c, la=1)
    rho = Co[0, 1] / Co[0, 0]  # lag one autocorrelation

    # if autocorrelation is like exp(- lag/lam) then, for lag = 1
    lam = -1.0 / log(abs(rho))

    # Our initial guess for maxlag is 1.5 times lam (eg. three times the mean
    # life)
    maxlag = int(floor(3.0 * lam)) + 1

    # We take 1% of lam to jump forward and look for the
    # rholimit threshold
    jmp = int(ceil(0.01 * lam)) + 1

    T = shape(Ser)[0]  # Number of rows in the matrix (sample size)

    while (abs(rho) > rholimit) & (maxlag < min(T / 2, maxmaxlag)):
        Co = AutoCov(Ser, c, la=maxlag)
        rho = Co[0, 1] / Co[0, 0]
        maxlag = maxlag + jmp
        ###print("maxlag=", maxlag, "rho", abs(rho), "\n")

    maxlag = int(floor(1.3 * maxlag))
    # 30% more

    if maxlag >= min(T / 2, maxmaxlag):  # not enough data
        fixmaxlag = min(min(T / 2, maxlag), maxmaxlag)
        print(
            "AutoMaxlag: Warning: maxlag= %d > min(T/2,maxmaxlag=%d), fixing it to %d" % (maxlag, maxmaxlag, fixmaxlag)
        )
        return fixmaxlag

    if maxlag <= 1:
        fixmaxlag = 10
        print("AutoMaxlag: Warning: maxlag= %d ?!, fixing it to %d" % (maxlag, fixmaxlag))
        return fixmaxlag

    print("AutoMaxlag: maxlag= %d." % maxlag)
    return maxlag


# Find the IAT
def IAT(Ser, cols=-1, maxlag=0, start=0, end=0):
    ncols = shape(mat(cols))[1]  # Number of columns to analyse (parameters)
    if ncols == 1:
        if cols == -1:
            cols = shape(Ser)[1] - 1  # default = last column
        cols = [cols]

    if end == 0:
        end = shape(Ser)[0]

    if maxlag == 0:
        for c in cols:
            maxlag = max(maxlag, AutoMaxlag(Ser[start:end, :], c))

    # print("IAT: Maxlag=", maxlag)

    # Ga = MakeSumMat(maxlag) * AutoCorr( Ser[start:end,:], cols=cols, la=maxlag)

    Ga = mat(zeros((maxlag / 2, ncols)))
    auto = AutoCorr(Ser[start:end, :], cols=cols, la=maxlag)

    # Instead of producing the maxlag/2 X maxlag MakeSumMat matrix, we
    # calculate the gammas like this
    for c in range(ncols):
        for i in range(maxlag / 2):
            Ga[i, c] = auto[2 * i, c] + auto[2 * i + 1, c]

    cut = Cutts(Ga)
    nrows = shape(Ga)[0]

    ncols = shape(cut)[1]
    Out = -1.0 * mat(ones([1, ncols]))

    if any((cut + 1) == nrows):
        print("IAT: Warning: Not enough lag to calculate IAT")

    for c in range(ncols):
        for i in range(cut[0, c] + 1):
            Out[0, c] += 2 * Ga[i, c]

    return Out
