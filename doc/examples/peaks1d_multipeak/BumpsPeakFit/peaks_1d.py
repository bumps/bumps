"""
Author: Paul Kienzle NIST NCNR
Modified by: Andrew Caruana STFC ISIS
For 1d peak fitting
"""

from __future__ import division, print_function

import warnings
from math import radians, sin, cos, sqrt, pi
from scipy.special import voigt_profile
import numpy as np

from bumps.parameter import Parameter, varying
from .errors import get_intervals, interval_low_high_from_interval


def _plot_view(view):
    import pylab
    
    if view == 'log':
        pylab.xscale('linear')
        pylab.yscale('log')
    elif view == 'logx':
        pylab.xscale('log')
        pylab.yscale('linear')
    elif view == 'logy':
        pylab.xscale('linear')
        pylab.yscale('log')
    elif view == 'loglog':
        pylab.xscale('log')
        pylab.yscale('log')
    else: 
        pylab.xscale('linear')
        pylab.yscale('linear')


def plot(X, theory, data, err, peaks=None, view=None):
    import pylab

    # vmin = np.amin(data)
    # vmax = np.amax(data)
    # window = 0.2*(vmax - vmin)
    pylab.subplot(211)
    pylab.errorbar(X, data, err)
    pylab.plot(X, theory)
    pylab.ylabel(r"Counts", fontsize=21)
    pylab.yticks(fontsize=18)
    # pylab.tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom="off")
    pylab.xticks(fontsize=0.01)

    if peaks is not None:
        try:
            for peak in peaks:
                pylab.plot(X, peak)
        except ValueError:
            # for now we just dont plot the background since it is a float not array
            # - ideally the background should also return an array
            pass
    _plot_view(view)
    pylab.subplot(212)
    pylab.scatter(X, (data-theory) / err)
    pylab.ylabel("Residuals\n" + r"$\frac{data - theory}{err}$", fontsize=21)
    pylab.yticks(fontsize=18)
    pylab.xlabel(r"Q ($\AA^{-1}$)", fontsize=21)
    pylab.xticks(fontsize=18)


def _calc_intervals(errs, intervals):

    total_array, peaks_array, peak_names = errs
    # caclulate intervals
    total_intv = []
    peaks_dict = {}
    for intv in intervals:
        total_intv.append(get_intervals(total_array, intv))
    for peak, peak_name in zip(peaks_array, peak_names):
        peaks_intv = []
        for intv in intervals:
            peaks_intv.append(get_intervals(peak, intv))
        peaks_dict[peak_name] = peaks_intv

    return total_intv, peaks_dict


class Gaussian(object):
    def __init__(self, A=1, xc=0, sig=1, name=""):
        self.name = name
        self.A = Parameter(A, name=name+"-A")
        self.xc = Parameter(xc, name=name+"-xc")
        self.sig = Parameter(sig, name=name+"-sig")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc,
                    sig=self.sig,
                    )
    
    @staticmethod
    def gauss_calc(x, xc, sigma, A):

        return A*np.exp(-((x-xc)**2/(2*sigma**2)))

    def __call__(self, x):
        amplitude = self.A.value
        sigma = self.sig.value
        xc = self.xc.value

        return self.gauss_calc(x, xc, sigma, amplitude)

    def __str__(self):
        return self.name


class Voigt(object):
    def __init__(self, A=1, xc=0, sig=1, gam=1, name=""):
        self.name = name
        self.A = Parameter(A, name=name+"-A")
        self.xc = Parameter(xc, name=name+"-xc")
        self.sig = Parameter(sig, name=name+"-sig")
        self.gam = Parameter(gam, name=name+"-gam")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc,
                    sig=self.sig,
                    gam=self.gam
                    )
    
    @staticmethod
    def voigt_calc(x, xc, sigma, gamma, amp):
        x_input = x - xc
        return amp*voigt_profile(x_input, sigma, gamma)

    def __call__(self, x):
        amplitude = self.A.value
        sigma = self.sig.value
        gamma = self.gam.value
        xc = self.xc.value

        return self.voigt_calc(x, xc, sigma, gamma, amplitude)

    def __str__(self):
        return self.name


class Background(object):
    def __init__(self, C=0, name=""):
        self.name = name
        self.C = Parameter(C, name=name+"-background")

    def parameters(self):
        return dict(C=self.C)

    def __call__(self, x):
        c = self.C.value
        return np.ones(x.shape) * c

    def __str__(self):
        return self.name


class Peaks(object):
    def __init__(self, parts, X, data, err, plot_peaks=None):
        self.X, self.data, self.err = X, data, err
        self.parts = parts
        self.plot_peaks = plot_peaks

    def numpoints(self):
        return np.prod(self.data.shape)

    def parameters(self):
        return [p.parameters() for p in self.parts]

    def theory(self):
        # return self.parts[0](self.X,self.Y)
        # parts = [M(self.X,self.Y) for M in self.parts]
        # for i,p in enumerate(parts):
        #    if np.any(np.isnan(p)): print "NaN in part",i
        return sum(M(self.X) for M in self.parts)

    def parts_theory(self):
        """
        returns a list of theory calculations for each part (peaks+backgrounds)
        which can be passed to the plotter
        """
        return [M(self.X) for M in self.parts]

    def residuals(self):
        # if np.any(self.err ==0): print "zeros in err"
        return (self.theory()-self.data) / self.err

    def nllf(self):
        R = self.residuals()
        # if np.any(np.isnan(R)): print "NaN in residuals"
        return 0.5*np.sum(R**2)

    # __call__ is never used. Also self.dof is not defined.
    # def __call__(self):
    #     return 2*self.nllf()/self.dof

    def plot(self, view='linear'):
        if self.plot_peaks is None:
            peaks = None
        else:
            peaks = self.parts_theory()
            # print(peaks[0])

        plot(self.X, self.theory(), self.data, self.err, peaks, view)

    def save(self, basename):
        import json
        pars = [(p.name, p.value) for p in varying(self.parameters())]
        out = json.dumps(dict(theory=self.theory().tolist(),
                              data=self.data.tolist(),
                              err=self.err.tolist(),
                              X=self.X.tolist(),
                              pars=pars))
        open(basename+".json", "w").write(out)

    def update(self):
        pass

    def calc_forwardmc(self, problem, points):
        # this should loop over points using the method of
        # add_forwardmc in each loop. By being in fitness
        # we can control how the fitness object handles the output
        # from add_forwardmc
        # TODO: add in handling for MultiFitProblem
        #  Need to decide how to handle the plots in this case,
        #  as the interface will get cluttered quickly
        if hasattr(problem, 'models'):
            warnings.warn("MultiFitProblem is not yet implemented for uncertainty plots")
            return
        # set the best point
        best = points[-1]
        problem.setp(best)
        # calculate the theory from add_forwardmc
        best_total, best_peaks = self.add_forwardmc()

        # calculate the peaks and total theory for the samples/points
        peak_names = [part.name for part in self.parts]
        peaks = []
        total = []
        for pts in points:
            problem.setp(pts)
            peaks.append(self.parts_theory())
            total.append(self.theory())
        peaks_array = np.transpose(np.array(peaks), axes=(1, 0, 2))
        total_array = np.array(total)
        return total_array, peaks_array, peak_names, best_peaks, best_total

    def plot_forwardmc(self, errors, intervals=(68, 95), save=None):
        # should take the role of show errors
        import matplotlib.pyplot as plt
        from itertools import cycle
        import pylab

        # fig, ax = plt.subplots()
        plt.subplot(111)
        colours = cycle(['r', 'b', 'g', 'c', 'm'])
        # TODO: add total theory plot method
        # total
        # p_68, p_95 = total
        # ax.fill_between(M.X,*p_68, color='r')
        # ax.fill_between(M.X,*p_95, alpha=0.5, color='r')
        total_array, peaks_array, peak_names, best_peaks, best_total = errors
        x, data = self.X, self.data
        total, peaks_dict = _calc_intervals((total_array, peaks_array, peak_names), intervals)

        # TODO: maybe place the plot method below it its own method?
        # for all peaks, without total theory
        for i, (peak_name, peak) in enumerate(peaks_dict.items()):
            p_68, p_95 = peak
            colour = next(colours)
            pylab.fill_between(x, *p_68, alpha=0.5, color=colour)
            pylab.fill_between(x, *p_95, alpha=0.25, color=colour)

            pylab.plot(x, best_peaks[i], color=colour, label=f"{peak_name} Best")

        plt.plot(x, best_total, color='k', label="Best")
        plt.scatter(x, data, s=3, color='k', label="Data")
        
        plt.ylabel(r"Counts", fontsize=21)
        plt.yticks(fontsize=18)
        plt.xlabel(r"Q ($\AA^{-1}$)", fontsize=21)
        plt.xticks(fontsize=18)
        
        plt.legend()
        # plt.draw()

        if save:
            print("trying to save model uncertainty")
            pylab.savefig(save + "-err.png")

    def save_forwardmc(self, errors, intervals=(68, 95), save=None):

        total_array, peaks_array, peak_names, best_peaks, best_total = errors
        x, data = self.X, self.data
        total, peaks_dict = _calc_intervals((total_array, peaks_array, peak_names), intervals)
        peaks_dict["total"] = total
        best_peaks.append(best_total)

        intervals_l_h = []
        for inter in intervals:
            intervals_l_h.extend(interval_low_high_from_interval(inter))
        k = 1
        for i, (peak_name, peak) in enumerate(peaks_dict.items()):
            save_data = np.vstack((x, best_peaks[i], *peak))
            columns = ["q", "best"] + list("%g%%" % (v*100) for v in intervals_l_h)
            self._write_file(save+"_contour%d.dat" % k, save_data, peak_name, columns)
            k += 1

    @staticmethod
    def _write_file(path, data, title, columns):
        with open(path, "wb") as fid:
            fid.write(b"# " + bytes(title, 'utf-8') + b"\n")
            fid.write(b"# " + bytes(" ".join(columns), 'utf-8') + b"\n")
            np.savetxt(fid, data.T)

    def add_forwardmc(self):
        # calculate the theory curves
        total = self.theory()
        peaks = self.parts_theory()
        return total, peaks

    def _find_by_name(self, target):
        """
        Iterate over all layers that have the given name.
        Code modified from refl1d.model.Stack()
        """
        for i, part in enumerate(self.parts):
            if str(part) == target:
                yield self.parts, i

    def _lookup(self, idx):
        """
        Lookup a part (peak or Background) by integer index or name. Search is from nth part first.
        Returns the parts list and the index of the part.
        Code modified from refl1d.model.Stack()
        """
        if isinstance(idx, int):
            return self.parts, idx

        # Check for lookup of the nth occurrence of a given part (if there is more than one of the same name)
        if isinstance(idx, tuple):
            target, count = idx
        else:
            target, count = idx, 1

        # Check if lookup by name
        if isinstance(target, str):
            sequence = self._find_by_name(target)
        else:
            raise TypeError("expected integer, peak name as sample index")

        # Move to the nth item in the sequence
        i = -1
        for i, el in enumerate(sequence):
            if i+1 == count:
                return el
        if i == -1:
            raise IndexError("part %s not found" % str(target))
        else:
            raise IndexError(f"only found {i+1} layers of {target}")

    def __getitem__(self, idx):
        parts, idx = self._lookup(idx)

        return parts[idx]


# ========================================================================================================
# 2D fitting classes and helper functions - lightly modified from peaks.py example from bumps. Not Tested!
# ========================================================================================================

def plot2d(X, Y, theory, data, err, view=None):
    import pylab

    # Not sure if 2d plots need different views.
    # put this as a place holder if required in future.
    if view is not None:
        raise NotImplementedError

    # print("theory",theory[1:6,1:6])
    # print("data",data[1:6,1:6])
    # print("delta",(data-theory)[1:6,1:6])
    vmin = np.amin(data)
    vmax = np.amax(data)
    window = 0.2*(vmax - vmin)
    pylab.subplot(131)
    pylab.pcolormesh(X, Y, data, vmin=vmin-window, vmax=vmax+window)
    pylab.subplot(132)
    pylab.pcolormesh(X, Y, theory, vmin=vmin-window, vmax=vmax+window)
    pylab.subplot(133)
    pylab.pcolormesh(X, Y, (data-theory)/(err+1))


class Cauchy2d(object):
    r"""
    2-D Cauchy

    https://en.wikipedia.org/wiki/Cauchy_distribution#Multivariate_Cauchy_distribution
    """
    def __init__(self, A=1, xc=0, yc=0, g1=1, g2=1, theta=0, name=""):
        self.A = Parameter(A, name=name+"A")
        self.xc = Parameter(xc, name=name+"xc")
        self.yc = Parameter(yc, name=name+"yc")
        self.g1 = Parameter(g1, name=name+"g1")
        self.g2 = Parameter(g2, name=name+"g2")
        self.theta = Parameter(theta, name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    g1=self.g1, g2=self.g2,
                    theta=self.theta)

    def __call__(self, x, y):
        area = self.A.value
        g1 = self.g1.value
        g2 = self.g2.value
        t = radians(self.theta.value)
        xc = self.xc.value
        yc = self.yc.value
        xbar, ybar = x-xc, y-yc
        a = cos(t)**2/g1**2 + sin(t)**2/g2**2
        b = sin(2*t)*(-1/g1**2 + 1/g2**2)
        c = sin(t)**2/g1**2 + cos(t)**2/g2**2
        gsq = a*xbar**2 + b*xbar*ybar + c*ybar**2
        Zf = 1./(2*pi*sqrt(g1*g2)*(1 + gsq)**1.5)
        # return Zf*abs(area)
        total = np.sum(Zf)
        return Zf/total*abs(area) if total > 0 else np.zeros_like(x)


class Lorentzian2d(object):
    r"""
    Lorentzian peak.

    Note that this is not equivalent to the multidimensional Cauchy
    distribution which models the sum of parameters as having a cauchy
    distribution.  Instead, it sets the gamma parameter according to
    elliptical direction
    sum
    """
    def __init__(self, A=1, xc=0, yc=0, g1=1, g2=1, theta=0, name=""):
        self.A = Parameter(A, name=name+"A")
        self.xc = Parameter(xc, name=name+"xc")
        self.yc = Parameter(yc, name=name+"yc")
        self.g1 = Parameter(g1, name=name+"g1")
        self.g2 = Parameter(g2, name=name+"g2")
        self.theta = Parameter(theta, name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    g1=self.g1, g2=self.g2,
                    theta=self.theta)

    def __call__(self, x, y):
        area = self.A.value
        g1 = self.g1.value
        g2 = self.g2.value
        t = radians(self.theta.value)
        xc = self.xc.value
        yc = self.yc.value
        # shift and rotate
        x, y = x-xc, y-yc
        x, y = x*cos(t) + y*sin(t), -x*sin(t) + y*cos(t)
        Zf = cauchy(x, g1) * cauchy(y, g2)
        # return Zf*abs(area)
        total = np.sum(Zf)
        return Zf/total*abs(area) if total > 0 else np.zeros_like(x)


class Voigt2d(object):
    r"""
    Voigt peak
    """
    def __init__(self, A=1, xc=0, yc=0, s1=1, s2=1, g1=1, g2=1, theta=0, name=""):
        self.A = Parameter(A, name=name+"A")
        self.xc = Parameter(xc, name=name+"xc")
        self.yc = Parameter(yc, name=name+"yc")
        self.s1 = Parameter(s1, name=name+"s1")
        self.s2 = Parameter(s2, name=name+"s2")
        self.g1 = Parameter(g1, name=name+"g1")
        self.g2 = Parameter(g2, name=name+"g2")
        self.theta = Parameter(theta, name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    s1=self.s1, s2=self.s2,
                    g1=self.g1, g2=self.g2,
                    theta=self.theta)

    def __call__(self, x, y):
        area = self.A.value
        s1 = self.s1.value
        s2 = self.s2.value
        g1 = self.g1.value
        g2 = self.g2.value
        t = radians(self.theta.value)
        xc = self.xc.value
        yc = self.yc.value
        # shift and rotate
        x, y = x-xc, y-yc
        x, y = x*cos(t) + y*sin(t), -x*sin(t) + y*cos(t)
        Zf = voigt(x, s1, g1) * voigt(y, s2, g2)
        # return Zf*abs(area)
        total = np.sum(Zf)
        return Zf/total*abs(area) if total > 0 else np.zeros_like(x)


class Background2d(object):
    def __init__(self, C=0, name=""):
        self.C = Parameter(C, name=name+"background")

    def parameters(self):
        return dict(C=self.C)

    def __call__(self, x, y):
        return self.C.value


class Peaks2d(object):
    def __init__(self, parts, X, Y, data, err):
        self.X, self.Y, self.data, self.err = X, Y, data, err
        self.parts = parts

    def numpoints(self):
        return np.prod(self.data.shape)

    def parameters(self):
        return [p.parameters() for p in self.parts]

    def theory(self):
        # return self.parts[0](self.X,self.Y)
        # parts = [M(self.X,self.Y) for M in self.parts]
        # for i,p in enumerate(parts):
        #    if np.any(np.isnan(p)): print "NaN in part",i
        return sum(M(self.X, self.Y) for M in self.parts)

    def residuals(self):
        # if np.any(self.err ==0): print "zeros in err"
        return (self.theory()-self.data)/(self.err+(self.err == 0.))

    def nllf(self):
        R = self.residuals()
        # if np.any(np.isnan(R)): print "NaN in residuals"
        return 0.5*np.sum(R**2)

    # __call__ is never used. Also self.dof is not defined
    # def __call__(self):
    #     return 2*self.nllf()/self.dof

    def plot(self, view=None):
        plot2d(self.X, self.Y, self.theory(), self.data, self.err, view=view)

    def save(self, basename):
        import json
        pars = [(p.name, p.value) for p in varying(self.parameters())]
        out = json.dumps(dict(theory=self.theory().tolist(),
                              data=self.data.tolist(),
                              err=self.err.tolist(),
                              X=self.X.tolist(),
                              Y=self.Y.tolist(),
                              pars=pars))
        open(basename+".json", "w").write(out)

    def update(self):
        pass


def cauchy(x, gamma):
    return gamma/(x**2 + gamma**2)/pi


def gauss(x, sigma):
    return np.exp(-0.5*(x/sigma)**2)/np.sqrt(2*pi*sigma**2)


def voigt(x, sigma, gamma):
    """
    Return the voigt function, which is the convolution of a Lorentz
    function with a Gaussian.

    :Parameters:
     gamma : real
      The half-width half-maximum of the Lorentzian
     sigma : real
      The 1-sigma width of the Gaussian, which is one standard deviation.

    Ref: W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64

    Note: adjusted to use stddev and HWHM rather than FWHM parameters
    """
    # wofz function = w(z) = Fad[d][e][y]eva function = exp(-z**2)erfc(-iz)
    from scipy.special import wofz
    z = (x+1j*gamma)/(sigma*np.sqrt(2))
    V = wofz(z)/(np.sqrt(2*pi)*sigma)
    return V.real
