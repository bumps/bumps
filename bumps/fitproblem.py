"""
Interface between the models and the fitters.

:class:`Fitness` defines the interface that model evaluators can follow.
These models can be bundled together into a :func:`FitProblem` and sent
to :class:`bumps.fitters.FitDriver` for optimization and uncertainty
analysis.
"""
# Don't include print_function in imports; since the model coded is exec'd
# in the __future__ context of this file, it would force the models to use the
# new print function syntax.  load_problem() should be moved to its own file
# to avoid this issue.
from __future__ import division, with_statement

__all__ = ['Fitness', 'FitProblem', 'load_problem',
           'BaseFitProblem', 'MultiFitProblem']

import sys
import os
import traceback
import logging

import numpy as np
from numpy import inf, isnan, NaN

from . import parameter, bounds as mbounds
from .formatnum import format_uncertainty
from . import util

# Abstract base class
class Fitness(object):
    """
    Manage parameters, data, and theory function evaluation.

    See :ref:`fitness` for a detailed explanation.
    """
    def parameters(self):
        """
        Return the parameters in the model.

        Model parameters are a hierarchical structure of lists and
        dictionaries.
        """
        raise NotImplementedError

    def update(self):
        """
        Called when parameters have been updated.  Any cached values will need
        to be cleared and the model reevaluated.
        """
        raise NotImplementedError

    def numpoints(self):
        """
        Return the number of data points.
        """
        raise NotImplementedError

    def nllf(self):
        """
        Return the negative log likelihood value of the current parameter set.
        """
        raise NotImplementedError

    def resynth_data(self):
        """
        Generate fake data based on uncertainties in the real data.  For
        Monte Carlo resynth-refit uncertainty analysis.  Bootstrapping?
        """
        raise NotImplementedError

    def restore_data(self):
        """
        Restore the original data in the model (after resynth).
        """
        raise NotImplementedError

    def residuals(self):
        """
        Return residuals for current theory minus data.

        Used for Levenburg-Marquardt, and for plotting.
        """
        raise NotImplementedError

    def save(self, basename):
        """
        Save the model to a file based on basename+extension.  This will point
        to a path to a directory on a remote machine; don't make any
        assumptions about information stored on the server.  Return the set of
        files saved so that the monitor software can make a pretty web page.
        """
        pass

    def plot(self, view='linear'):
        """
        Plot the model to the current figure.  You only get one figure, but you
        can make it as complex as you want.  This will be saved as a png on
        the server, and composed onto a results web page.
        """
        pass


def no_constraints():
    """default constraints function for FitProblem"""
    return 0


# TODO: refactor FitProblem definition
# deprecate the direct use of MultiFitProblem
def FitProblem(*args, **kw):
    """
    Return a fit problem instance for the fitness function(s).

    For an individual model:

        *fitness* is a :class:`Fitness` instance.

    For a set of models:

        *models* is a sequence of :class:`Fitness` instances.

        *weights* is an optional scale factor for each model

        *freevars* is :class:`parameter.FreeVariables` instance defining the
        per-model parameter assignments.  See :ref:`freevariables` for details.


    Additional parameters:

        *name* name of the problem

        *constraints* is a function which returns the negative log likelihood
        of seeing the parameters independent from the fitness function.  Use
        this for example to check for feasible regions of the search space, or
        to add constraints that cannot be easily calculated per parameter.
        Ideally, the constraints nllf will increase as you go farther from
        the feasible region so that the fit will be directed toward feasible
        values.

        *soft_limit* is the constraints function cutoff, beyond which the
        *penalty_nllf* will be used and *fitness* nllf will not be calculated.

        *penalty_nllf* is the nllf to use for *fitness* when *constraints*
        is greater than *soft_limit*.

    Total nllf is the sum of the parameter nllf, the constraints nllf and the
    depending on whether constraints is greater than soft_limit, either the
    fitness nllf or the penalty nllf.
    """
    if len(args) > 0:
        try:
            models = list(args[0])
        except TypeError:
            models = args[0]
        if isinstance(models, list):
            return MultiFitProblem(models, *args[1:], **kw)
        else:
            return BaseFitProblem(*args, **kw)
    else:
        if 'fitness' in kw:
            return BaseFitProblem(*args, **kw)
        else:
            return MultiFitProblem(*args, **kw)


class BaseFitProblem(object):
    """
    See :func:`FitProblem`
    """
    def __init__(self, fitness, name=None, constraints=no_constraints,
                 penalty_nllf=1e6, soft_limit=np.inf, partial=False):
        self.constraints = constraints
        self.fitness = fitness
        self.partial = partial
        if name is not None:
            self.name = name
        else:
            try:
                self.name = fitness.name
            except AttributeError:
                self.name = 'FitProblem'

        self.soft_limit = soft_limit
        self.penalty_nllf = penalty_nllf
        self.model_reset()

    # noinspection PyAttributeOutsideInit
    def model_reset(self):
        """
        Prepare for the fit.

        This sets the parameters and the bounds properties that the
        solver is expecting from the fittable object.  We also compute
        the degrees of freedom so that we can return a normalized fit
        likelihood.

        If the set of fit parameters changes, then model_reset must
        be called.
        """
        # print self.model_parameters()
        all_parameters = parameter.unique(self.model_parameters())
        # print "all_parameters",all_parameters
        self._parameters = parameter.varying(all_parameters)
        # print "varying",self._parameters
        self.bounded = [p for p in all_parameters
                        if not isinstance(p.bounds, mbounds.Unbounded)]
        self.dof = self.model_points()
        if not self.partial:
            self.dof -= len(self._parameters)
        if self.dof <= 0:
            raise ValueError("Need more data points than fitting parameters")
        #self.constraints = pars.constraints()

    def model_parameters(self):
        """
        Parameters associated with the model.
        """
        return self.fitness.parameters()

    def model_points(self):
        """
        Number of data points associated with the model.
        """
        return self.fitness.numpoints()

    def model_update(self):
        """
        Update the model according to the changed parameters.
        """
        if hasattr(self.fitness, 'update'):
            self.fitness.update()

    def model_nllf(self):
        """
        Negative log likelihood of seeing data given model.
        """
        return self.fitness.nllf()

    def simulate_data(self, noise=None):
        """Simulate data with added noise"""
        self.fitness.simulate_data(noise=noise)

    def resynth_data(self):
        """Resynthesize data with noise from the uncertainty estimates."""
        self.fitness.resynth_data()

    def restore_data(self):
        """Restore original data after resynthesis."""
        self.fitness.restore_data()

    def valid(self, pvec):
        """Return true if the point is in the feasible region"""
        return all(v in p.bounds for p, v in zip(self._parameters, pvec))

    def setp(self, pvec):
        """
        Set a new value for the parameters into the model.  If the model
        is valid, calls model_update to signal that the model should be
        recalculated.

        Returns True if the value is valid and the parameters were set,
        otherwise returns False.
        """
        # TODO: do we have to leave the model in an invalid state?
        # WARNING: don't try to conditionally update the model
        # depending on whether any model parameters have changed.
        # For one thing, the model_update below probably calls
        # the subclass MultiFitProblem.model_update, which signals
        # the individual models.  Furthermore, some parameters may
        # related to others via expressions, and so a dependency
        # tree needs to be generated.  Whether this is better than
        # clicker() from SrFit I do not know.
        for v, p in zip(pvec, self._parameters):
            p.value = v
        # TODO: setp_hook is a hack to support parameter expressions in sasview
        # Don't depend on this existing long term.
        setp_hook = getattr(self, 'setp_hook', no_constraints)
        setp_hook()
        self.model_update()

    def getp(self):
        """
        Returns the current value of the parameter vector.
        """
        return np.array([p.value for p in self._parameters], 'd')

    def bounds(self):
        """Return the bounds fore each parameter a 2 x N array"""
        return np.array([p.bounds.limits for p in self._parameters], 'd').T

    def randomize(self, n=None):
        """
        Generates a random model.

        *randomize()* sets the model to a random value.

        *randomize(n)* returns a population of *n* random models.

        For indefinite bounds, the random population distribution is centered
        on initial value of the parameter, or 1. if the initial parameter is
        not finite.
        """
        # TODO: split into two: randomize and random_pop
        if n is None:
            self.setp(self.randomize(n=1)[0])
            return   # Not returning anything since no n is requested

        target = self.getp()
        target[~np.isfinite(target)] = 1.
        pop = [p.bounds.random(n, target=v)
               for p, v in zip(self._parameters, target)]
        return np.array(pop).T

    def parameter_nllf(self):
        """
        Returns negative log likelihood of seeing parameters p.
        """
        s = sum(p.nllf() for p in self.bounded)
        # print "; ".join("%s %g %g"%(p,p.value,p.nllf()) for p in
        # self.bounded)
        return s

    def constraints_nllf(self):
        """
        Returns the cost of all constraints.
        """
        return self.constraints()

    def parameter_residuals(self):
        """
        Returns negative log likelihood of seeing parameters p.
        """
        return [p.residual() for p in self.bounded]

    def residuals(self):
        """
        Return the model residuals.
        """
        return self.fitness.residuals()

    def chisq(self):
        """
        Return sum squared residuals normalized by the degrees of freedom.

        In the context of a composite fit, the reduced chisq on the individual
        models only considers the points and the fitted parameters within
        the individual model.

        Note that this does not include cost factors due to constraints on
        the parameters, such as sample_offset ~ N(0,0.01).
        """
        return np.sum(self.residuals() ** 2) / self.dof
        # return 2*self.nllf()/self.dof

    def chisq_str(self):
        """
        Return a string representing the chisq equivalent of the nllf.

        If the model has strictly gaussian independent uncertainties then the
        negative log likelihood function will return 0.5*sum(residuals**2),
        which is 1/2*chisq.  Since we are printing normalized chisq, we
        multiply the model nllf by 2/DOF before displaying the value.  This
        is different from the problem nllf function, which includes the
        cost of the prior parameters and the cost of the penalty constraints
        in the total nllf.  The constraint value is displayed separately.
        """
        pparameter, pconstraints, pmodel = self._nllf_components()
        chisq_norm, chisq_err = nllf_scale(self)
        chisq = pmodel * chisq_norm
        text = format_uncertainty(chisq, chisq_err)
        constraints = pparameter + pconstraints
        if constraints > 0.:
            text += " constraints=%g" % constraints

        return text

    def nllf(self, pvec=None):
        """
        Compute the cost function for a new parameter set p.

        This is not simply the sum-squared residuals, but instead is the
        negative log likelihood of seeing the data given the model parameters
        plus the negative log likelihood of seeing the model parameters.  The
        value is used for a likelihood ratio test so normalization constants
        can be ignored.  There is an additional penalty value provided by
        the model which can be used to implement inequality constraints.  Any
        penalty should be large enough that it is effectively excluded from
        the parameter space returned from uncertainty analysis.

        The model is not actually calculated if the parameter nllf plus the
        constraint nllf are bigger than *soft_limit*, but instead it is
        assigned a value of *penalty_nllf*.  This will prevent expensive
        models from spending time computing values in the unfeasible region.
        """
        if pvec is not None:
            if self.valid(pvec):
                self.setp(pvec)
            else:
                return inf

        pparameter, pconstraints, pmodel = self._nllf_components()
        cost = pparameter + pconstraints + pmodel
        # print(pvec, "cost=",pparameter,"+",pconstraints,"+",pmodel,"=",cost)
        if isnan(cost):
            # TODO: make sure errors get back to the user
            # print "point evaluates to NaN"
            # print parameter.summarize(self._parameters)
            return inf
        return cost

    def _nllf_components(self):
        try:
            pparameter = self.parameter_nllf()
            if isnan(pparameter):
                # TODO: make sure errors get back to the user
                info = ["Parameter nllf is wrong"]
                info += ["%s %g"%(p, p.nllf()) for p in self.bounded]
                logging.error("\n  ".join(info))
            pconstraints = self.constraints_nllf()
            pmodel = (self.model_nllf()
                      if pparameter + pconstraints <= self.soft_limit
                      else self.penalty_nllf)
            return pparameter, pconstraints, pmodel
        except Exception:
            # TODO: make sure errors get back to the user
            info = (traceback.format_exc(),
                    parameter.summarize(self._parameters))
            logging.error("\n".join(info))
            return NaN, NaN, NaN

    def __call__(self, pvec=None):
        """
        Problem cost function.

        Returns the negative log likelihood scaled by DOF so that
        the result looks like the familiar normalized chi-squared.  These
        scale factors will not affect the value of the minimum, though some
        care will be required when interpreting the uncertainty.
        """
        return 2 * self.nllf(pvec) / self.dof

    def show(self, _subs={}):
        """Print the available parameters to the console as a tree."""
        print(parameter.format(self.model_parameters(), freevars=_subs))
        print("[chisq=%s, nllf=%g]" % (self.chisq_str(), self.nllf()))
        #print(self.summarize())

    def summarize(self):
        """Return a table of current parameter values with range bars."""
        return parameter.summarize(self._parameters)

    def labels(self):
        """Return the list of labels, one per fitted parameter."""
        return [p.name for p in self._parameters]

    def save(self, basename):
        """
        Save the problem state for the current parameter set.

        The underlying Fitness object *save* method is called, if it exists,
        so that theory values can be saved in a format suitable to the problem.

        Uses *basename* as the base of any files that are created.
        """
        if hasattr(self.fitness, 'save'):
            self.fitness.save(basename)

    def plot(self, p=None, fignum=None, figfile=None, view=None):
        """
        Plot the problem state for the current parameter set.

        The underlying Fitness object *plot* method is called with *view*.
        It should produce its plot on the current matplotlib figure.  This
        method will add chisq to the plot and save it to a file.
        """
        if not hasattr(self.fitness, 'plot'):
            return

        import pylab
        if fignum is not None:
            pylab.figure(fignum)
        if p is not None:
            self.setp(p)
        self.fitness.plot(view=view)
        pylab.text(0.01, 0.01, 'chisq=%s' % self.chisq_str(),
                   transform=pylab.gca().transAxes)
        if figfile is not None:
            pylab.savefig(figfile + "-model.png", format='png')

    def cov(self):
        """
        Return the covariance matrix as computed by numdifftools from the
        Hessian matrix for the problem at the current parameter values.
        """
        from . import lsqerror
        H = lsqerror.hessian(self)
        H, L = lsqerror.perturbed_hessian(H)
        return lsqerror.chol_cov(L)

    def stderr(self):
        """
        Return the 1-sigma uncertainty estimate for each parameter and the
        correlation matrix *R* as computed from the covariance returned by
        *cov*.
        """
        from . import lsqerror
        c = self.cov()
        return lsqerror.stderr(c), lsqerror.corr(c)

    def __getstate__(self):
        return (self.fitness, self.partial, self.name, self.penalty_nllf,
                self.soft_limit, self.constraints)

    def __setstate__(self, state):
        self.fitness, self.partial, self.name, self.penalty_nllf, \
            self.soft_limit, self.constraints = state
        self.model_reset()

class MultiFitProblem(BaseFitProblem):
    """
    Weighted fits for multiple models.
    """
    def __init__(self, models, weights=None, name=None,
                 constraints=no_constraints,
                 soft_limit=np.inf, penalty_nllf=1e6,
                 freevars=None):
        self.partial = False
        self.constraints = constraints
        if freevars is None:
            names = ["M%d" % i for i, _ in enumerate(models)]
            freevars = parameter.FreeVariables(names=names)
        self.freevars = freevars
        self._models = [BaseFitProblem(m, partial=True) for m in models]
        if weights is None:
            weights = [1 for _ in models]
        self.weights = weights
        self.penalty_nllf = penalty_nllf
        self.soft_limit = soft_limit
        self.set_active_model(0)  # Set the active model to model 0
        self.model_reset()
        self.name = name

    @property
    def models(self):
        """Iterate over models, with free parameters set from model values"""
        for i, f in enumerate(self._models):
            self.freevars.set_model(i)
            yield f
        # Restore the active model after cycling
        self.freevars.set_model(self._active_model_index)

    # noinspection PyAttributeOutsideInit
    def set_active_model(self, i):
        """Use free parameters from model *i*"""
        self._active_model_index = i
        self.active_model = self._models[i]
        self.freevars.set_model(i)

    def model_parameters(self):
        """Return parameters from all models"""
        pars = {'models': [f.model_parameters() for f in self.models]}
        free = self.freevars.parameters()
        if free:
            pars['freevars'] = free
        return pars

    def model_points(self):
        """Return number of points in all models"""
        return sum(f.model_points() for f in self.models)

    def model_update(self):
        """Let all models know they need to be recalculated"""
        # TODO: consider an "on changed" signal for model updates.
        # The update function would be associated with model parameters
        # rather than always recalculating everything.  This
        # allows us to set up fits with 'fast' and 'slow' parameters,
        # where the fit can quickly explore a subspace where the
        # computation is cheap before jumping to a more expensive
        # subspace.  SrFit does this.
        for f in self.models:
            f.model_update()

    def model_nllf(self):
        """Return cost function for all data sets"""
        return sum(f.model_nllf() for f in self.models)

    def constraints_nllf(self):
        """Return the cost function for all constraints"""
        return (sum(f.constraints_nllf() for f in self.models)
                + BaseFitProblem.constraints_nllf(self))

    def simulate_data(self, noise=None):
        """Simulate data with added noise"""
        for f in self.models:
            f.simulate_data(noise=noise)

    def resynth_data(self):
        """Resynthesize data with noise from the uncertainty estimates."""
        for f in self.models:
            f.resynth_data()

    def restore_data(self):
        """Restore original data after resynthesis."""
        for f in self.models:
            f.restore_data()

    def residuals(self):
        resid = np.hstack([w * f.residuals()
                           for w, f in zip(self.weights, self.models)])
        return resid

    def save(self, basename):
        for i, f in enumerate(self.models):
            f.save(basename + "-%d" % (i + 1))

    def show(self):
        for i, f in enumerate(self.models):
            print("-- Model %d %s" % (i, f.name))
            subs = self.freevars.get_model(i) if self.freevars else {}
            f.show(_subs=subs)
        print("[overall chisq=%s, nllf=%g]" % (self.chisq_str(), self.nllf()))

    def plot(self, p=None, fignum=1, figfile=None, view=None):
        import pylab
        if p is not None:
            self.setp(p)
        for i, f in enumerate(self.models):
            f.plot(fignum=i + fignum, view=view)
            pylab.suptitle('Model %d - %s' % (i, f.name))
            if figfile is not None:
                pylab.savefig(figfile + "-model%d.png" % i, format='png')

    # Note: restore default behaviour of getstate/setstate rather than
    # inheriting from BaseFitProblem
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# TODO: consider adding nllf_scale to FitProblem.
ONE_SIGMA = 0.68268949213708585
def nllf_scale(problem):
    r"""
    Return the scale factor for reporting the problem nllf as an approximate
    normalized chisq, along with an associated "uncertainty".  The uncertainty
    is the amount that chisq must change in order for the fit to be
    significantly better.

    From Numerical Recipes 15.6: *Confidence Limits on Estimated Model
    Parameters*, the $1-\sigma$ contour in parameter space corresponds
    to $\Delta\chi^2 = \text{invCDF}(1-\sigma,k)$ where
    $1-\sigma \approx 0.6827$ and $k$ is the number of fitting parameters.
    Since we are reporting the normalized $\chi^2$, this needs to be scaled
    by the problem degrees of freedom, $n-k$, where $n$ is the number of
    measurements.  To first approximation, the uncertainty in $\chi^2_N$
    is $k/(n-k)$
    """
    dof = getattr(problem, 'dof', np.NaN)
    if dof <= 0 or np.isnan(dof) or np.isinf(dof):
        return 1., 0.
    else:
        #return 2./dof, 1./dof
        from scipy.stats import chi2
        npars = max(len(problem.getp()), 1)
        return 2./dof, chi2.ppf(ONE_SIGMA, npars)/dof

def load_problem(filename, options=None):
    """
    Load a problem definition from a python script file.

    sys.argv is set to ``[file] + options`` within the context of the script.

    The user must define ``problem=FitProblem(...)`` within the script.

    Raises ValueError if the script does not define problem.
    """
    # Allow relative imports from the bumps model
    package = util.relative_import(filename)
    module = os.path.splitext(os.path.basename(filename))[0]

    ctx = dict(__file__=filename, __package__=package, __name__=module)
    old_argv = sys.argv
    sys.argv = [filename] + options if options else [filename]
    source = open(filename).read()
    code = compile(source, filename, 'exec')
    exec(code, ctx)
    sys.argv = old_argv
    try:
        problem = ctx["problem"]
    except KeyError:
        raise ValueError(filename + " requires 'problem = FitProblem(...)'")

    return problem
