"""
Interface between the models and the fitters.

:class:`Fitness` defines the interface that model evaluators can follow.
These models can be bundled together into a :func:`FitProblem` and sent
to :class:`bumps.fitters.FitDriver` for optimization and uncertainty
analysis.


Summary of problem attributes::

    # Used by fitters
    nllf(p: Optional[Vector]) -> float  # main calculation
    bounds() -> Tuple(Vector, Vector)    # or equivalent sequence
    setp(p: Vector) -> None
    getp() -> Vector
    residuals() -> Vector  # for LM, MPFit
    parameter_residuals() -> Vector  # for LM, MPFit
    constraints_nllf() -> float # for LM, MPFit;  constraint cost is spread across the individual residuals
    randomize() -> None # for multistart
    resynth_data() -> None  # for Monte Carlo resampling of maximum likelihood
    restore_data() -> None # for Monte Carlo resampling of maximum likelihood
    name: str  # DREAM uses this
    chisq() -> float
    chisq_str() -> str
    labels() -> List[str]
    summarize() -> str
    show() -> None
    load(input_path: str) -> None
    save(output_path: str) -> None
    plot(figfile: str, view: str) -> None

    # Set/used by bumps.cli
    model_reset() -> None # called by load_model
    path: str  # set by load_model
    name: str # set by load_model
    title: str = filename # set by load_moel
    options: List[str]  # from sys.argv[1:]
    undefined:List[int]  # when loading a save .par file, these parameters weren't defined
    store: str # set by make_store
    output_path: str # set by make_store
    simulate_data(noise: float) -> None # for --simulate in opts
    cov() -> Matrix   # for --cov in opts

"""

__all__ = ["Fitness", "FitProblem", "load_problem"]

from dataclasses import dataclass
import logging
import os
import sys
import traceback
from typing import Generic, TypeVar
import warnings

import numpy as np
from numpy import inf, isnan, nan

from . import parameter, util
from .parameter import to_dict, Parameter, Variable, tag_all
from .formatnum import format_uncertainty


# Abstract base class:
# can use "isinstance" to check if a class implements the protocol
@util.runtime_checkable
@dataclass(init=False)
class Fitness(util.Protocol):
    """
    Manage parameters, data, and theory function evaluation.

    See :ref:`fitness` for a detailed explanation.
    """

    def parameters(self) -> util.List[Parameter]:
        """
        return the parameters in the model.

        model parameters are a hierarchical structure of lists and
        dictionaries.
        """
        raise NotImplementedError()

    def update(self):
        """
        Called when parameters have been updated.  Any cached values will need
        to be cleared and the model reevaluated.
        """
        raise NotImplementedError()

    def numpoints(self):
        """
        Return the number of data points.
        """
        raise NotImplementedError()

    def nllf(self):
        """
        Return the negative log likelihood value of the current parameter set.
        """
        raise NotImplementedError()

    def resynth_data(self):
        """
        Generate fake data based on uncertainties in the real data.  For
        Monte Carlo resynth-refit uncertainty analysis.  Bootstrapping?
        """
        raise NotImplementedError()

    def restore_data(self):
        """
        Restore the original data in the model (after resynth).
        """
        raise NotImplementedError()

    def residuals(self):
        """
        Return residuals for current theory minus data.

        Used for Levenburg-Marquardt, and for plotting.
        """
        raise NotImplementedError()

    def save(self, basename):
        """
        Save the model to a file based on basename+extension.  This will point
        to a path to a directory on a remote machine; don't make any
        assumptions about information stored on the server.  Return the set of
        files saved so that the monitor software can make a pretty web page.
        """
        pass

    def plot(self, view="linear"):
        """
        Plot the model to the current figure.  You only get one figure, but you
        can make it as complex as you want.  This will be saved as a png on
        the server, and composed onto a results web page.
        """
        pass


def no_constraints() -> float:
    """default constraints function for FitProblem"""
    return 0


def fit_parameters(fitness: Fitness) -> util.List[Parameter]:
    """
    Return a list of fittable (non-fixed) parameters in the model
    """
    parameters = parameter.unique(fitness.parameters())
    return [p for p in parameters if isinstance(getattr(p, "slot", None), parameter.Variable) and not p.fixed]


def chisq_str(fitness: Fitness) -> str:
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

    pars = fit_parameters(fitness)
    dof = fitness.numpoints() - len(pars)
    if dof <= 0 or np.isnan(dof) or np.isinf(dof):
        chisq_norm, chisq_err = 1.0, 0.0
    else:
        # return 2./dof, 1./dof
        from scipy.stats import chi2

        npars = max(len(pars), 1)
        chisq_norm, chisq_err = 2.0 / dof, chi2.ppf(ONE_SIGMA, npars) / dof

    chisq = fitness.nllf() * chisq_norm
    text = format_uncertainty(chisq, chisq_err)
    return text


def show_parameters(fitness: Fitness, subs: util.Optional[util.Dict[util.Any, Parameter]] = None):
    """Print the available parameters to the console as a tree."""
    print(parameter.format(fitness.parameters(), freevars=subs))
    print("[chisq=%s, nllf=%g]" % (chisq_str(fitness), fitness.nllf()))


FitnessType = TypeVar("FitnessType", bound=Fitness)


@dataclass(init=False, eq=False)
class FitProblem(Generic[FitnessType]):
    r"""

        *models* is a sequence of :class:`Fitness` instances.

        *weights* is an optional scale factor for each model. A weighted fit
        returns nllf $L = \sum w_k^2 L_k$. If an individual nllf is the sum
        squared residuals then this is equivalent to scaling the measurement
        uncertainty by $1/w$. Unless the measurement uncertainty is unknown,
        weights should be in [0, 1], representing an unknown systematic
        uncertainty spread across the individual measurements.

        *freevars* is :class:`.parameter.FreeVariables` instance defining the
        per-model parameter assignments.  See :ref:`freevariables` for details.


    Additional parameters:

        *name* name of the problem

        *constraints* is a list of Constraint objects, which have a method to
        calculate the nllf for that constraint.
        Also supports an alternate form which cannot be serialized:
        A function which returns the negative log likelihood
        of seeing the parameters independent from the fitness function.  Use
        this for example to check for feasible regions of the search space, or
        to add constraints that cannot be easily calculated per parameter.
        Ideally, the constraints nllf will increase as you go farther from
        the feasible region so that the fit will be directed toward feasible
        values.

        *penalty_nllf* is the nllf to use for *fitness* when *constraints*
        or model parameter bounds are not satisfied.

    Total nllf is the sum of the parameter nllf, the constraints nllf and the
    depending on whether constraints is greater than soft_limit, either the
    fitness nllf or the penalty nllf.

    New in 0.9.0: weights are now squared when computing the sum rather than
    linear.
    """

    name: util.Optional[str]
    models: util.List[FitnessType]
    freevars: util.Optional[parameter.FreeVariables]
    weights: util.Union[util.List[float], util.Literal[None]]
    constraints: util.Optional[util.Sequence[parameter.Constraint]]
    penalty_nllf: util.Union[float, util.Literal["inf"]]

    _constraints_function: util.Callable[..., float]
    _models: util.List[FitnessType]
    _parameters: util.List[Parameter]
    _parameters_by_id: util.Dict[str, Parameter]
    _dof: float = np.nan  # not a schema field, and is not used in __init__

    # _all_constraints: util.List[util.Union[Parameter, Expression]]

    def __init__(
        self,
        models: util.Union[FitnessType, util.List[FitnessType]],
        weights=None,
        name=None,
        constraints=None,
        penalty_nllf="inf",
        freevars=None,
        soft_limit: util.Optional[float] = None,  # TODO: deprecate,
        auto_tag=False,
    ):
        if not isinstance(models, (list, tuple)):
            models = [models]
        if callable(constraints):
            warnings.warn("convert constraints functions to constraint expressions", DeprecationWarning)
            self._constraints_function = constraints
            self.constraints = []
        else:
            self._constraints_function = self._null_constraints_function
            # TODO: do we want to allow "constraints=a<b" or do we require a sequence "constraints=[a<b]"?
            self.constraints = constraints if constraints is not None else []
        if freevars is None:
            names = ["M%d" % i for i, _ in enumerate(models)]
            freevars = parameter.FreeVariables(names=names)
        self.freevars = freevars
        if auto_tag:
            for index, model in enumerate(models):
                model_name = model.name if model.name is not None else f"Model{index}"
                tag_all(model.parameters(), model_name)
        self._models = models
        if weights is None:
            weights = [1.0 for _ in models]
        self.weights = weights
        self.penalty_nllf = float(penalty_nllf)
        self.set_active_model(0)  # Set the active model to model 0
        self.model_reset()  # sets self._all_constraints
        self.name = name

    @staticmethod
    def _null_constraints_function():
        return 0.0

    @property
    def fitness(self):
        warnings.warn("Deprecated: use of problem.fitness will be removed at some point")
        if len(self._models) == 1:
            return self._models[0]
        raise ValueError("problem.fitness is not defined")

    @property
    def dof(self):
        return self._dof

    # TODO: make this @property\ndef models(self): ...
    @property
    def models(self):
        """Iterate over models, with free parameters set from model values"""
        try:
            for i, f in enumerate(self._models):
                self.freevars.set_model(i)
                yield f
        finally:
            # Restore the active model after cycling, even if interrupted
            self.freevars.set_model(self._active_model_index)

    # noinspection PyAttributeOutsideInit
    def set_active_model(self, i):
        """Use free parameters from model *i*"""
        self._active_model_index = i
        self.active_model = self._models[i]
        self.freevars.set_model(i)

    def model_parameters(self):
        """Return parameters from all models"""
        pars = {}
        pars["models"] = [f.parameters() for f in self.models]
        free = self.freevars.parameters()
        if free:
            pars["freevars"] = free
        return pars

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "name": self.name,
            "models": to_dict(self._models),
            "weights": self.weights,
            "penalty_nllf": self.penalty_nllf,
            # TODO: constraints may be a function.
            "constraints": to_dict(self.constraints),
            "freevars": to_dict(self.freevars),
        }

    def __repr__(self):
        return "FitProblem(name=%s)" % self.name

    def valid(self, pvec):
        """Return true if the point is in the feasible region"""
        return all(v in p.prior for p, v in zip(self._parameters, pvec))

    def setp(self, pvec):
        """
        Set a new value for the parameters into the model.  If the model
        is valid, calls model_update to signal that the model should be
        recalculated.

        Returns True if the value is valid and the parameters were set,
        otherwise returns False.
        """
        # print("Calling setp with", pvec, self._parameters)
        # TODO: do we have to leave the model in an invalid state?
        # WARNING: don't try to conditionally update the model
        # depending on whether any model parameters have changed.
        # For one thing, the model_update below probably calls
        # the subclass FitProblem.model_update, which signals
        # the individual models.  Furthermore, some parameters may
        # related to others via expressions, and so a dependency
        # tree needs to be generated.  Whether this is better than
        # clicker() from SrFit I do not know.
        for v, p in zip(pvec, self._parameters):
            p.value = v
        # TODO: setp_hook is a hack to support parameter expressions in sasview
        # Don't depend on this existing long term.
        setp_hook = getattr(self, "setp_hook", no_constraints)
        setp_hook()
        self.model_update()

    def getp(self):
        """
        Returns the current value of the parameter vector.
        """
        return np.array([p.value for p in self._parameters], "d")

    def bounds(self):
        """Return the bounds for each parameter as a 2 x N array"""
        limits = [p.prior.limits for p in self._parameters]
        return np.array(limits, "d").T if limits else np.empty((2, 0))

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
            return  # Not returning anything since no n is requested

        # TODO: apply hard limits on parameters
        target = self.getp()
        target[~np.isfinite(target)] = 1.0
        pop = [p.prior.random(n, target=v) for p, v in zip(self._parameters, target)]
        return np.array(pop).T

    def nllf(self, pvec=None) -> float:
        """
        compute the cost function for a new parameter set p.

        this is not simply the sum-squared residuals, but instead is the
        negative log likelihood of seeing the data given the model parameters
        plus the negative log likelihood of seeing the model parameters.  the
        value is used for a likelihood ratio test so normalization constants
        can be ignored.  there is an additional penalty value provided by
        the model which can be used to implement inequality constraints.  any
        penalty should be large enough that it is effectively excluded from
        the parameter space returned from uncertainty analysis.

        the model is not actually calculated if any of the parameters are
        out of bounds, or any of the constraints are not satisfied, but
        instead are assigned a value of *penalty_nllf*.
        this will prevent expensive models from spending time computing
        values in the unfeasible region.
        """
        if pvec is not None:
            if self.valid(pvec):
                self.setp(pvec)
            else:
                return inf

        pparameter, pconstraints, pmodel, failing_constraints = self._nllf_components()
        cost = pparameter + pconstraints + pmodel
        # print("cost=",pparameter,"+",pconstraints,"+",pmodel,"=",cost, pvec)
        if isnan(cost):
            # TODO: make sure errors get back to the user
            # print "point evaluates to nan"
            # print parameter.summarize(self._parameters)
            return inf
        return cost

    def parameter_nllf(self) -> util.Tuple[float, util.List[str]]:
        """
        Returns negative log likelihood of seeing parameters p.
        """
        failing = []
        nllf = 0.0
        for p in self._bounded:
            p_nllf = p.nllf()
            nllf += p_nllf
            if p_nllf == np.inf:
                failing.append(str(p))
        # print "; ".join("%s %g %g"%(p,p.value,p.nllf()) for p in
        # self._bounded)
        return nllf, failing

    def parameter_residuals(self):
        """
        Returns negative log likelihood of seeing parameters p.
        """
        return [p.residual() for p in self._bounded]

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
        pparameter, pconstraints, pmodel, failing_constraints = self._nllf_components()
        chisq_norm, chisq_err = nllf_scale(self)
        chisq = pmodel * chisq_norm
        text = format_uncertainty(chisq, chisq_err)
        constraints = pparameter + pconstraints
        if constraints > 0.0:
            text += " constraints=%g" % constraints
        if len(failing_constraints) > 0:
            text += " failing_constraints=%s" % str(failing_constraints)

        return text

    def _nllf_components(self) -> util.Tuple[float, float, float, util.List[str]]:
        try:
            pparameter, failing_parameter_constraints = self.parameter_nllf()
            if isnan(pparameter):
                # TODO: make sure errors get back to the user
                info = ["Parameter nllf is wrong"]
                info += ["%s %g" % (p, p.nllf()) for p in self._bounded]
                logging.error("\n  ".join(info))
            pconstraints, failing_constraints = self.constraints_nllf()
            failing_constraints += failing_parameter_constraints
            # Note: for hard constraints (which return inf) avoid computing
            # model even if soft_limit is inf by using strict comparison
            # since inf <= inf is True but inf < inf is False.
            penalty_nllf = self.penalty_nllf if self.penalty_nllf is not None else np.inf
            pmodel = self.model_nllf() if len(failing_constraints) == 0 else penalty_nllf

            return pparameter, pconstraints, pmodel, failing_constraints
        except Exception:
            # TODO: make sure errors get back to the user
            info = (traceback.format_exc(), parameter.summarize(self._parameters))
            logging.error("\n".join(info))
            return nan, nan, nan, []

    def __call__(self, pvec=None):
        """
        Problem cost function.

        Returns the negative log likelihood scaled by DOF so that
        the result looks like the familiar normalized chi-squared.  These
        scale factors will not affect the value of the minimum, though some
        care will be required when interpreting the uncertainty.
        """
        return 2 * self.nllf(pvec) / self.dof

    def summarize(self):
        """Return a table of current parameter values with range bars."""
        return parameter.summarize(self._parameters)

    def labels(self) -> util.List[str]:
        """Return the list of labels, one per fitted parameter."""
        return [p.name for p in self._parameters]

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
        # print("In model reset with", self.model_parameters())
        all_parameters = parameter.unique(self.model_parameters())
        # print "all_parameters",all_parameters
        # for p in all_parameters:
        #     if hasattr(p, 'reset_prior'):
        #         p.reset_prior()  # no constraints
        #     else:
        #         raise ValueError(f"{p} does not have prior")
        broken = []
        for p in all_parameters:
            # slot = p.slot
            # value = p.value

            # TODO: this is a shim to accomodate Expression, Calculation etc. being
            # put into attributes that have type "Parameter" (in user scripts)
            # Do we cause those scripts to break instead?
            # Or do we autoconvert to .equals()?
            if hasattr(p, "add_prior"):
                p.add_prior(p.distribution, bounds=p.bounds, limits=p.limits)

            # TODO: currently this logic for showing breaking constraints to the user
            # is in the chisq_str execution path instead of on model_reset - should it be here instead?

            # While we are walking all parameters check which constraints aren't satisfied
            # Build up a list of strings to help the user initialize the model correctly
            if hasattr(p, "limits"):
                value = p.value
                if (p.limits[0] > value) or (value > p.limits[1]):
                    broken.append(f"{p}={value} is outside {p.limits}")
                elif not np.isfinite(p.prior.nllf(value)):
                    broken.append(f"{p}={value} is outside {p.prior}")

        broken.extend([f"constraint {c} is unsatisfied" for c in self.constraints if float(c) == inf])
        if self._constraints_function() == inf:
            broken.append("user constraint function is unsatisfied")
        if len(broken) > 0:
            warnings.warn("Unsatisfied constraints: [%s]" % (",\n".join(broken)))
        self.broken_constraints = broken

        # TODO: shimmed to allow non-Parameter in Parameter attribute spots.
        pars = []
        pars_by_id = {}
        for p in all_parameters:
            if hasattr(p, "id"):
                pars_by_id[p.id] = p
            slot = getattr(p, "slot", None)
            if isinstance(slot, Variable) and not p.fixed:
                pars.append(p)
        self._parameters = pars
        self._parameters_by_id = pars_by_id
        # TODO: shimmed to allow non-Parameter in Parameter attribute spots.
        self._bounded = [p for p in all_parameters if hasattr(p, "has_prior") and p.has_prior()]
        self._dof = self.model_points()
        self._dof -= len(self._parameters)
        if self.dof <= 0:
            warnings.warn(
                f"Need more data points (currently: {self.model_points()}) than fitting parameters ({len(self._parameters)})"
            )
        # self.constraints = pars.constraints()
        # Find the constraints on variables and expressions that we need to compute
        # parameter_constraints = [p.slot for p in all_parameters if isinstance(p.slot, Variable) and p.has_prior()]
        # expression_constraints = [p.slot for p in all_parameters if isinstance(p.slot, Expression) and p.has_prior()]
        # self._all_constraints = parameter_constraints + expression_constraints

    def model_points(self):
        """Return number of points in all models"""
        return sum(f.numpoints() for f in self.models)

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
            if hasattr(f, "update"):
                f.update()

    def model_nllf(self):
        """Return cost function for all data sets"""
        return sum(w**2 * f.nllf() for w, f in zip(self.weights, self.models))

    def constraints_nllf(self) -> util.Tuple[float, util.List[str]]:
        """Return the cost function for all constraints"""
        failing = []
        nllf = 0.0
        nllf = self._constraints_function()
        if nllf == np.inf:
            failing.append("user constraints function")
        for c in self.constraints:
            # TODO: convert to list of residuals for Levenberg-Marquardt
            c_nllf = float(c) ** 2
            nllf += c_nllf
            if c_nllf > 0:
                failing.append(str(c))

        return nllf, failing

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

    @property
    def has_residuals(self):
        """
        True if all underlying fitness functions define residuals.
        """
        return all(hasattr(f, "residuals") for f in self.models)

    def residuals(self):
        resid = np.hstack([w * f.residuals() for w, f in zip(self.weights, self.models)])
        return resid

    def save(self, basename):
        for i, f in enumerate(self.models):
            if hasattr(f, "save"):
                f.save(basename + "-%d" % (i + 1))

    def show(self):
        for i, f in enumerate(self.models):
            print("-- Model %d %s" % (i, getattr(f, "name", "")))
            subs = self.freevars.get_model(i) if self.freevars else {}
            show_parameters(f, subs=subs)
        print("[overall chisq=%s, nllf=%g]" % (self.chisq_str(), self.nllf()))

    def plot(self, p=None, fignum=1, figfile=None, view=None, model_indices=None):
        import pylab

        if p is not None:
            self.setp(p)
        for i, f in enumerate(self.models):
            if model_indices is not None and i not in model_indices:
                continue
            if not hasattr(f, "plot"):
                continue
            f.plot(view=view)
            pylab.figure(i + fignum)
            f.plot(view=view)
            pylab.suptitle("Model %d - %s" % (i, f.name))
            pylab.text(0.01, 0.01, "chisq=%s" % chisq_str(f), transform=pylab.gca().transAxes)
            if figfile is not None:
                pylab.savefig(figfile + "-model%d.png" % i, format="png")

    # Note: restore default behaviour of getstate/setstate rather than
    # inheriting from BaseFitProblem
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# TODO: consider adding nllf_scale to FitProblem.
ONE_SIGMA = 0.68268949213708585


def nllf_scale(problem: FitProblem):
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
    dof = problem.dof
    if dof <= 0 or np.isnan(dof) or np.isinf(dof):
        return 1.0, 0.0
    else:
        # return 2./dof, 1./dof
        from scipy.stats import chi2

        npars = max(len(problem.getp()), 1)
        return 2.0 / dof, chi2.ppf(ONE_SIGMA, npars) / dof


def load_problem(filename, options=None) -> FitProblem:
    """
    Load a problem definition from a python script file.

    sys.argv is set to ``[file] + options`` within the context of the script.

    The user must define ``problem=FitProblem(...)`` within the script.

    Raises ValueError if the script does not define problem.
    """
    # Allow relative imports from the bumps model
    module_name = os.path.splitext(os.path.basename(filename))[0]
    module = util.relative_import(filename, module_name=module_name)

    ctx = dict(__file__=filename, __package__=module, __name__=module_name)
    old_argv = sys.argv
    sys.argv = [filename] + options if options else [filename]
    source = open(filename).read()
    code = compile(source, filename, "exec")
    exec(code, ctx)
    sys.argv = old_argv
    problem = ctx.get("problem", None)
    if problem is None:
        raise ValueError(filename + " requires 'problem = FitProblem(...)'")

    return problem


def MultiFitProblem(*args, **kwargs) -> FitProblem:
    warnings.warn(DeprecationWarning("use FitProblem directly instead of MultiFitProblem"))
    return FitProblem(*args, **kwargs)


def test_weighting():
    class SimpleFitness(Fitness):
        def __init__(self, a=0.0, name="fit"):
            self.a = parameter.Parameter.default(a, name=name + " a")

        def parameters(self):
            return {"a": self.a}

        def numpoints(self):
            return 1

        def residuals(self):
            y, dy = 0, 1  # fit to constant 0 +/- 1
            return np.array([(self.a.value - y) / dy])

        def nllf(self):
            return sum(r**2 for r in self.residuals()) / 2

    weights = 2, 3
    models = [SimpleFitness(4.0), SimpleFitness(5.0)]
    problem = FitProblem(models, weights=weights)

    # Need to use problem.models to cycle through models in case FreeVariables is used in problem
    assert (problem.residuals() == np.hstack([w * M.residuals() for w, M in zip(weights, problem.models)])).all()
    assert problem.nllf() == sum(w**2 * M.nllf() for w, M in zip(weights, problem.models))
    assert problem.nllf() == sum(problem.residuals() ** 2) / 2
