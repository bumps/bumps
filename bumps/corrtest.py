"""
Residual correlation test.

INCOMPLETE UNUSED UNTESTED CODE

Returns the probability of seeing the given structure in the residuals.

The residuals are assumed to be e ~ N(0,1).  Gross deviations from this
form will already be accounted for in the chisq statistic, and can be
ignored here.  We do this by translating the residual into a z-scored
residual, which should be in

Protects against an incorrect estimation of the uncertainty in the data,
which is anyway already covered by the chisq cost function.
"""

from scipy import stats

def residual_nllf(v):
    # Normalize the scores so that we are assuming
    z = stats.zs(v)



def plot(self):
    pylab.subplot(313)
    Q,R = self.reflectivity()
    self.probe.plot_transform(theory=(Q,R),
                    substrate=self.sample[0].material,
                    surround=self.sample[-1].material)

# Experiments with different sorts of
def plot_transform(self, theory=None, substrate=None, surround=None):
    """
    Plot the Fresnel reflectivity associated with the probe.
    """
    import pylab
    if substrate is None and surround is None:
        raise TypeError("Fresnel reflectivity needs substrate or surround")
    F = self.fresnel(substrate=substrate,surround=surround)
    print "substrate",substrate
    #Qc = sqrt(16*pi*substrate)
    Qc = 0
    T = numpy.linspace(Qc,max(self.Q),len(self.Q))
    if hasattr(self,'R'):
        A = abs(numpy.fft.fft(numpy.interp(T,self.Q,self.R/F)))
        pylab.plot(T,A, '.')
    if theory is not None:
        Q,R = theory
        A = abs(numpy.fft.fft(numpy.interp(T,self.Q,self.R/F)))
        pylab.plot(T,A, hold=True)
    pylab.xlabel('z (Angstroms)')
    if substrate is None:
        name = "air:%s"%(surround.name)
    elif surround is None or isinstance(surround,Vacuum):
        name = substrate.name
    else:
        name = "%s:%s"%(substrate.name, surround.name)
    pylab.ylabel('FFT(R/R(%s))'%(name))

def plot_deriv(self, theory=None):
    import pylab
    scale = 1e8*self.Q**4
    #scale=1
    if hasattr(self, 'R'):
        d = deriv(self.Q, self.R, self.dR)
        pylab.plot(self.Q, d*scale, '.')
    if theory is not None:
        Q,R = theory
        dth = deriv(Q,R)
        pylab.plot(Q,dth*scale,hold=True)
    #pylab.plot(Q,(sign(d)-sign(dth))/2)
    pylab.xlabel('Q (inv Angstroms)')
    pylab.ylabel('R\' (100 Q)^4')

def deriv(Q, R, dR=None, width=5, degree=2):
    from .wsolve import wpolyfit
    d = numpy.empty_like(Q)
    if dR is None: dR = numpy.ones_like(Q)
    k = (width-1)/2
    p = wpolyfit(Q[:width],R[:width],dy=dR[:width],degree=degree)
    d[:k+1]=p.der(Q[:k+1])
    for i in range(k+1,len(Q)-k-1):
        idx = slice(i-k,i+k+1)
        p = wpolyfit(Q[idx],R[idx],dy=dR[idx],degree=degree)
        d[i]=p.der(Q[i])
    p = wpolyfit(Q[-width:],R[-width:],dy=dR[-width:],degree=degree)
    d[-k-1:] = p.der(Q[-k-1:])
    return d
