from __future__ import division, print_function

__all__ = [
    "Experiment",
    "load_data",
    "load_model",
    "load_fit",
    "sim_data",
    "Parameter",
    "FitProblem",
    "FreeVariables",
    "pmath",
    "preview",
    "fit",
    "np",
    "sys",
    "sans",
    "seed",
]

# symbols loaded for export
import sys
import numpy as np
from bumps.names import Parameter, FitProblem, FreeVariables, pmath, preview, fit
from bumps.parameter import Parameter
from bumps.util import push_seed as seed
import sans
import sans.models

# symbols needed internally
from sans.dataloader.data_info import Data1D, Data2D
from sans.fit.AbstractFitEngine import FitData1D, FitData2D
from sans.perspectives.fitting.pagestate import Reader as FitReader
from sans.dataloader.loader import Loader as DataLoader


def load_data(filename):
    return DataLoader().load(filename)


def sim_data(model, noise=5, qmin=0.005, qmax=0.5, nq=100, dq=0):
    for pid, p in getattr(model, "_bumps_pars", {}).items():
        model.setParam(pid, p.value)
    q = np.logspace(np.log10(qmin), np.log10(qmax), nq)
    # if dq != 0 then need smearing
    I = model.evalDistribution(q)
    dI = I * noise / 100.0
    I += np.random.randn(*q.shape) * dI
    return Data1D(x=q, dx=dq, y=I, dy=dI)


def load_model(model, name=None, **kw):
    sans = __import__("sans.models." + model)
    ModelClass = getattr(getattr(sans.models, model, None), model, None)
    if ModelClass is None:
        raise ValueError("could not find model %r in sans.models" % model)
    M = ModelClass()
    prefix = (name if name else _model_name(M)) + " "
    M._bumps_pars = {}
    valid_pars = M.getParamList()
    for k, v in kw.items():
        # dispersion parameters initialized with _field instead of .field
        if k.endswith("_width"):
            k = k[:-6] + ".width"
        elif k.endswith("_npts"):
            k = k[:-5] + ".npts"
        elif k.endswith("_nsigmas"):
            k = k[:-7] + ".nsigmas"
        elif k.endswith("_type"):
            k = k[:-5] + ".type"
        if k not in valid_pars:
            formatted_pars = ", ".join(valid_pars)
            raise KeyError("invalid parameter %r for %s--use one of: %s" % (k, model, formatted_pars))
        if "." in k and not k.endswith(".width"):
            M.setParam(k, v)
        elif isinstance(v, Parameter):
            M._bumps_pars[k] = v
        elif isinstance(v, (tuple, list)):
            low, high = v
            P = Parameter((low + high) / 2, bounds=v, name=prefix + k)
            M._bumps_pars[k] = P
        else:
            P = Parameter(v, name=prefix + k)
            M._bumps_pars[k] = P
    return M


def load_fit(filename):
    data = FitReader(call_back=lambda **kw: None).read("FitPage2.fitv")
    data = data[0]  # no support for multiset files
    fit = data.meta_data["fitstate"]
    model_name = fit.formfactorcombobox
    pars = dict((p[1], float(p[2])) for p in fit.parameters)
    for k, v in pars.items():
        if abs(v) < 1e-5 and v != 0:
            pars[k] = 1e-6 * Parameter(v * 1e6, name=model_name + " " + k)
    model = load_model(model_name, **pars)
    experiment = Experiment(model=model, data=data)
    for p in fit.parameters:
        if p[0]:
            low = float(p[5][1]) if p[5][1] else -np.inf
            high = float(p[6][1]) if p[6][1] else np.inf
            try:
                experiment[p[1]].range(low, high)
            except KeyError:
                print("%s not in experiment" % p[1])
    return experiment


def _model_name(model):
    name = model.__class__.__name__
    if name.endswith("Model"):
        name = name[:-5]
    return name.lower()


def _sas_parameter(model, pid, prefix):
    par = getattr(model, "_bumps_pars", {}).get(pid, None)
    if par is None:
        ## Don't have bounds on dispersion parameters with model details
        # bounds = model.details.get(pid, [None, None, None])[1:3]
        value = model.getParam(pid)
        par = Parameter(value, name=prefix + pid)
    return par


def _build_parameters(model, prefix, oriented, magnetic):
    # Gather the list of parameters, stripping out the distribution attributes
    pars = set(pid for pid in model.getParamList() if "." not in pid or pid.endswith(".width"))
    if not oriented:
        pars -= set(model.orientation_params)
    if not magnetic:
        pars -= set(model.magnetic_params)
    return dict((pid, _sas_parameter(model, pid, prefix)) for pid in pars)


def _set_parameters(model, pars):
    for pid, p in pars.items():
        # print("setting %r to %g"%(pid, p.value))
        model.setParam(pid, p.value)


class Experiment(object):
    def __init__(self, model, data, smearer=None, qmin=None, qmax=None, name=""):
        self.name = name if name else model.__class__.__name__
        self.model = model

        self.oriented = isinstance(data, Data2D)
        self.magnetic = False

        # Convert data to fitdata
        if self.oriented:
            self.fitdata = FitData2D(sans_data2d=data, data=data.data, err_data=data.err_data)
        else:
            self.fitdata = FitData1D(x=data.x, y=data.y, dx=data.dx, dy=data.dy, data=data)
        self.fitdata.sans_data = data
        self.fitdata.set_fit_range(qmin, qmax)
        # self.fitdata.set_smearer(smearer)

        # save some bits of info
        self._saved_y = self.fitdata.y
        prefix = (name if name else _model_name(model)) + " "
        self._pars = _build_parameters(model, prefix, self.oriented, self.magnetic)
        self._cache = {}

    def __getitem__(self, key):
        return self._pars[key]

    def __setitem__(self, key, value):
        self._pars[key] = value

    def theory(self):
        key = "theory"
        if key not in self._cache:
            _set_parameters(self.model, self._pars)
            resid, fx = self.fitdata.residuals(self.model.evalDistribution)
            self._cache[key] = fx
            self._cache["residuals"] = resid
        return self._cache[key]

    def parameters(self):
        """
        Return the set of parameters in the model.
        """
        return self._pars

    def update(self):
        """
        Called when parameters have been updated.  Any cached values will need to
        be cleared and the model reevaluated.
        """
        self._cache = {}

    def numpoints(self):
        """
        Return the number of data points.
        """
        return len(self.fitdata.x)

    def nllf(self):
        """
        Return the negative log likelihood value of the current parameter set.
        """
        return 0.5 * np.sum(self.residuals() ** 2)

    def resynth_data(self):
        """
        Generate fake data based on uncertainties in the real data.  For Monte Carlo
        resynth-refit uncertainty analysis.  Bootstrapping?
        """
        y, dy = self._saved_y, self.fitdata.dy
        self.data.y = y + np.random.randn(len(y)) * dy

    def restore_data(self):
        """
        Restore the original data in the model (after resynth).
        """
        self.data.y = self._saved_y

    def residuals(self):
        """
        Return residuals for current theory minus data.  For levenburg-marquardt.
        """
        self.theory()  # automatically calculates residuals
        return self._cache["residuals"]

    def save(self, basename):
        """
        Save the model to a file based on basename+extension.  This will point to
        a path to a directory on a remote machine; don't make any assumptions about
        information stored on the server.  Return the set of files saved so that
        the monitor software can make a pretty web page.
        """
        pass

    def plot(self, view=None):
        """
        Plot the model to the current figure.  You only get one figure, but you
        can make it as complex as you want.  This will be saved as a png on
        the server, and composed onto a results webpage.
        """
        # print("view", view)
        import pylab

        if self.oriented:
            qx, qy, Iqxy = self.fitdata.qx_data, self.fitdata.qy_data, self.fitdata.data
            xlabel, ylabel = self.fitdata.sans_data
            pylab.subplot(311)
            pylab.pcolormesh(qx, qy, Iqxy)
            pylab.title("Data")
            pylab.subplot(312)
            pylab.pcolormesh(qx, qy, self.theory())
            pylab.title("Theory")
            pylab.subplot(313)
            pylab.pcolormesh(qx, qy, self.residuals(), vmin=-3, vmax=3)
            pylab.title("Residuals +/- 3 sigma")
        elif view == "residual":
            pylab.plot(self.fitdata.x, self.residuals(), ".")
            pylab.axhline(1, color="black", ls="--", lw=1)
            pylab.axhline(0, color="black", lw=1)
            pylab.axhline(-1, color="black", ls="--", lw=1)
            pylab.xlabel("Q (inv A)")
            pylab.ylabel("(theory-data)/error")
            pylab.legend(numpoints=1)
        else:
            pylab.errorbar(
                self.fitdata.x,
                self.fitdata.y,
                xerr=self.fitdata.dx,
                yerr=self.fitdata.dy,
                fmt="o",
                label="data " + self.name,
            )
            pylab.plot(self.fitdata.x, self.theory(), "-", label="fit " + self.name)
            pylab.xscale("log")
            pylab.yscale("log")
