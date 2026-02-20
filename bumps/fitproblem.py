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

__all__ = ["Fitness", "FitProblem", "CovarianceMixin", "load_problem"]

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
import sys
import traceback
from typing import Generic, TypeVar, Union, Optional
import warnings
from pathlib import Path

import numpy as np
from numpy import inf, isnan, nan
from scipy.stats import chi2

from . import parameter, util
from .parameter import to_dict, Parameter, Variable, tag_all, priors
from .util import format_uncertainty


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
        pass

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


# The functions below would be methods of Fitness if it were a
# base class rather than a protocol.


def fitness_parameters(fitness: Fitness) -> util.List[Parameter]:
    """
    Return a list of fittable (non-fixed) parameters in the model
    """
    parameters = parameter.unique(fitness.parameters())
    return [p for p in parameters if isinstance(getattr(p, "slot", None), parameter.Variable) and not p.fixed]


def fitness_chisq(fitness: Fitness) -> str:
    """
    Return a string representing the chisq equivalent of the nllf for
    a single dataset. Unlike FitProblem.chisq_str, this does not
    include parameter uncertainty or constraint penalty.
    """
    npars = len(fitness_parameters(fitness))
    dof = fitness.numpoints() - npars
    chisq_norm, _ = nllf_scale(dof=dof, npars=npars, norm=True)

    # TODO: Check if parameters to fitness are in feasible region before computing nllf?
    chisq = fitness.nllf() * chisq_norm
    return chisq


def fitness_chisq_str(fitness: Fitness) -> str:
    """
    Return a string representing the chisq equivalent of the nllf for
    a single dataset. Unlike FitProblem.chisq_str, this does not
    include parameter uncertainty or constraint penalty.
    """

    npars = len(fitness_parameters(fitness))
    dof = fitness.numpoints() - npars
    chisq_norm, chisq_err = nllf_scale(dof=dof, npars=npars, norm=True)

    # TODO: Check if parameters are in feasible region before computing nllf?
    chisq = fitness.nllf() * chisq_norm
    text = format_uncertainty(chisq, chisq_err)
    # return f"{text} {fitness.nllf()=:.15e} {chisq_norm=:.15e} {chisq_err=:.15e} {dof=}"
    return text


def fitness_show_parameters(fitness: Fitness, subs: util.Optional[util.Dict[util.Any, Parameter]] = None):
    """Print the available parameters to the console as a tree."""
    print(parameter.format(fitness.parameters(), freevars=subs))
    print("[chisq=%s, nllf=%g]" % (fitness_chisq_str(fitness), fitness.nllf()))


FitnessType = TypeVar("FitnessType", bound=Fitness)


class CovarianceMixin:
    """
    Add methods for *cov*, *show_cov* and *show_err* to a bumps problem definition.

    This is done as a mixin because not all problems are FitProblem. See
    for example :class:`bumps.pdfwrapper.PDF`.
    """

    # Note: if caching is implemented for covariance, make sure it is cleared on setp
    # This will be difficult to do as a mixin.
    def cov(self, x):
        r"""
        Return an estimate of the covariance of the fit.

        Depending on the fitter and the problem, this may be computed from
        existing evaluations within the fitter, or from numerical
        differentiation around the minimum.

        If the problem has residuals available, then the covariance
        is derived from the Jacobian::

            x = fit.problem.getp()
            J = bumps.lsqerror.jacobian(fit.problem, x)
            cov = bumps.lsqerror.jacobian_cov(J)

        Otherwise, the numerical differentiation will use the Hessian
        estimated from nllf::

            x = fit.problem.getp()
            H = bumps.lsqerror.hessian(fit.problem, x)
            cov = bumps.lsqerror.hessian_cov(H)
        """
        # Use Jacobian if residuals are available because it is faster
        # to compute.  Otherwise punt and use Hessian.  The has_residuals
        # attribute should be True if present.  It may be false if
        # the problem defines a residuals method but doesn't really
        # have residuals (e.g. to allow levenberg-marquardt to run even
        # though it is not fitting a sum-square problem).
        from bumps import lsqerror

        if hasattr(self, "has_residuals"):
            has_residuals = self.has_residuals
        else:
            has_residuals = hasattr(self, "residuals")

        if has_residuals:
            J = lsqerror.jacobian(self, x)
            # print("Jacobian", J)
            return lsqerror.jacobian_cov(J)
        else:
            H = lsqerror.hessian(self, x)
            # print("Hessian", H)
            return lsqerror.hessian_cov(H)

    def show_cov(self, x, cov):
        maxn = 1000  # max array dims to print
        cov_str = np.array2string(
            cov,
            max_line_width=20 * maxn,
            threshold=maxn * maxn,
            precision=6,  # suppress_small=True,
            separator=", ",
        )
        print("=== Covariance matrix ===")
        print(cov_str)
        print("=========================")

    def show_err(self, x, dx):
        """
        Display the error approximation from the covariance matrix.

        *err* is the standard deviation computed from the covariance matrix. It
        is available as *result.dx* from the simple fitter, or using::

            from bumps import lsqerror

            dx = lsqerror.stderr(problem.cov(x))

        Warning: cost to compute cov grows as the cube of the number of parameters.
        """
        # TODO: need cheaper uncertainty estimate
        # Note: error estimated from hessian diagonal is insufficient.
        print("=== Uncertainty from curvature:     name   value(unc.) ===")
        for k, v, dv in zip(self.labels(), x, dx):
            print(f"{k:>40s}   {format_uncertainty(v, dv):<15s}")
        print("=" * 58)


# The default penalty nllf has to be big enough that it won't be swamped by
# the nllf of the residuals even for a 100 Mpixel image, but not so
# big that the distance to the boundary is not lost in the floating point precision.
PENALTY_NLLF = 1e12


# TODO: add filename to fitproblem so we don't have to coordinate it elsewhere?
@dataclass(init=False, eq=False)
class FitProblem(Generic[FitnessType], CovarianceMixin):
    r"""

        *models* is a sequence of :class:`Fitness` instances. Note that they
        do not need to all be of the same class.

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
        or model parameter bounds are not satisfied. The total nllf is the
        squared distance from the boundary plus the penalty so that the derivative
        points the search back to the feasible region. The penalty should be larger
        than any nllf you might see near the boundary so that the fit doesn't get
        stuck outside, but small enough that penalty plus distance is different from
        penalty. The default is 1e12.

    Total nllf is the sum of the parameter nllf, the constraints nllf and the
    depending on whether constraints is greater than soft_limit, either the
    fitness nllf or the penalty nllf.

    New in 0.9.0: weights are now squared when computing the sum rather than
    linear.
    """

    # TODO: problem.path is set by cli.load_model(); should we add it as standard?
    name: util.Optional[str]
    models: util.List[FitnessType]
    freevars: util.Optional[parameter.FreeVariables]
    weights: util.Union[util.List[float], util.Literal[None]]
    constraints: util.Optional[util.Sequence[parameter.Constraint]]
    penalty_nllf: util.Union[float, util.Literal["inf"]]

    # The type is not quite correct. The models do not need to be of the same class
    _models: util.List[FitnessType]
    _priors: util.List[Parameter]
    _parameters: util.List[Parameter]
    _parameters_by_id: util.Dict[str, Parameter]
    _dof: float = np.nan  # not a schema field, and is not used in __init__
    _active_model_index: int = 0
    # _all_constraints: util.List[util.Union[Parameter, Expression]]

    def __init__(
        self,
        models: util.Union[FitnessType, util.List[FitnessType]],
        weights=None,
        name=None,
        constraints=None,
        penalty_nllf=None,
        freevars=None,
        auto_tag=False,
    ):
        if not isinstance(models, (list, tuple)):
            models = [models]
        if callable(constraints):
            raise TypeError("Callable constraints function is no longer supported. Instead use a list of comparisons.")
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
        self.penalty_nllf = float(penalty_nllf) if penalty_nllf is not None else PENALTY_NLLF
        self.name = name

        # Do these steps last so that it has all of the attributes initialized.
        self.model_reset()  # sets self._all_constraints
        self.set_active_model(0)  # Set the active model to model 0

    @property
    def fitness(self):
        warnings.warn("Deprecated: use of problem.fitness will be removed at some point")
        if len(self._models) == 1:
            return self._models[0]
        raise ValueError("problem.fitness is not defined")

    @property
    def dof(self):
        return self._dof

    @property
    def num_models(self):
        return len(self._models)

    @property
    def models(self):
        """Iterate over models, with free parameters set from model values"""
        try:
            for index in range(len(self._models)):
                index, model = self._switch_model(index)
                yield model
        finally:
            # Restore the active model after cycling, even if interrupted
            self._switch_model(self._active_model_index)

    # TODO: deprecate set_active_model and push_model
    # set_active_model is only used once in the wx gui. push_model is only needed
    # if we have a notion of active model.

    # noinspection PyAttributeOutsideInit
    def set_active_model(self, index):
        """
        Fetch model *index* with the appropriate free variables substituted.

        This will remain the active model until a new active model is selected.

        Operations like chisq_str() or plot() which cycle through the models will
        restore the parameters upon completion.
        """
        index, model = self._switch_model(index)
        self._active_model_index, self.active_model = index, model
        return model

    @contextmanager
    def push_model(self, index):
        """
        Fetch model *index* with the appropriate free variables substituted.

        On completion of the context, restore the parameters for the active model.
        """
        try:
            index, model = self._switch_model(index)
            yield model
        finally:
            # Restore the active model after cycling, even if interrupted
            self._switch_model(self._active_model_index)

    def _switch_model(self, index):
        # print(f"switching to {index} with freevars={bool(self.freevars)}")
        if not (-len(self._models) <= index < len(self._models)):
            raise IndexError(f"Index {index} invalid when only {len(self._models)} models")
        if index < 0:
            index = len(self._models) + index
        # TODO: FreeVariables destroys caching within the model. Replace it.
        # TODO: Only update parameters if index has changed. We can track this in freevars.
        self.freevars.set_model(index)
        if self.freevars:
            # Clear model cache when updating the parameters
            getattr(self._models[index], "update", lambda: None)()
        return index, self._models[index]

    def model_parameters(self):
        """Return parameters from all models"""
        pars = {}
        # Note: the self.models iterator plugs the free variables into
        # the model in turn, so no need to walk self.freevars directly.
        # The model.update() function is called each time, so whatever
        # caching is happening in the model is cleared, and it knows that
        # new parameter values have been inserted.
        pars["models"] = [f.parameters() for f in self.models]
        free = self.freevars.parameters()
        if free:
            pars["freevars"] = free
        return pars

    # TODO: no longer used?
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
        Compute the cost function for a new parameter set p.

        This is not simply the sum-squared residuals, but instead is the
        negative log likelihood of seeing the data given the model parameters
        plus the negative log likelihood of seeing the model parameters.  The
        value is used for a likelihood ratio test so normalization constants
        can be ignored.  There is an additional penalty value provided by
        the model which can be used to implement inequality constraints.  Any
        penalty should be large enough that it is effectively excluded from
        the parameter space returned from uncertainty analysis.

        The model is not actually calculated if any of the parameters are
        out of bounds, or any of the constraints are not satisfied, but
        instead are assigned a value of *penalty_nllf*. This will prevent
        expensive models from spending time computing values in the unfeasible
        region.
        """
        if pvec is not None:
            # Note that valid() only checks that the fit parameters are in the bounding box.
            # It doesn't check that all the constraints are satisfied. To do that we would
            # have to use setp on the vector then loop over p._priors, resetting to the
            # original pvec if the new pvec is invalid.
            if self.valid(pvec):
                self.setp(pvec)
            else:
                return inf

        pparameter, pconstraints, pmodel, failing = self._nllf_components()
        # Note: pmodel is zero if any constraints are failing. In that case the cost
        # will be the squared distance from the boundary of the feasible region for
        # the breaking parameters so that gradient descent can guide us back to the
        # feasible region. The penalty would ideally be greater than any value of
        # pmodel inside the boundary (otherwise the fitter may prefer the point in the
        # infeasible region) but it has to be small enough that adding a small distance
        # changes the penalty value (otherwise the slope outside the boundary is zero).
        # Currently using 1e12 as the default penalty. This will be too small for many problems
        # but any larger and the derivatives will break.
        # TODO: Drop the penalty term, either by rewriting the fitters so they know that
        # the constraints are broken, or rewriting the constraints to simple box constraints.
        cost = pparameter + pconstraints + (self.penalty_nllf if failing else pmodel)
        # print(f"prior:{float(pparameter)} + constraint:{float(pconstraints)} + nllf:{float(pmodel)} => {float(cost)} at [{pvec=}]")
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
        for p in self._priors:
            p_nllf = p.nllf()
            nllf += p_nllf
            if p_nllf == np.inf:
                failing.append(str(p))
        # print("; ".join(f"{p}({p.value})={p.nllf()}" for p in self._priors))
        return nllf, failing

    def parameter_residuals(self):
        """
        Returns negative log likelihood of seeing parameters p.
        """
        return [p.residual() for p in self._priors]

    def chisq(self, nllf: Union[float, util.NDArray, None] = None, norm: bool = True, compact: bool = True):
        """
        Returns chisq as a floating point value.

        See documentation for :meth:`chisq_str`.
        """
        chisq_norm, chisq_err = nllf_scale(dof=self.dof, npars=len(self._parameters), norm=norm)
        if nllf is None:
            pparameter, pconstraints, pmodel, failing = self._nllf_components()
            nllf = pmodel + pparameter + pconstraints + (self.penalty_nllf if failing else 0)
        return nllf * chisq_norm

    # TODO: Too many versions of chisq about.
    # Note: norm and compact are no longer used in bumps, so they are not documented
    def chisq_str(self, nllf: Optional[float] = None, norm: bool = True, compact: bool = True):
        """
        Return a string representing the chisq equivalent of the nllf.

        If *nllf* is provided then use that instead of calling the model
        evaluator. Fail if *compact* is False.

        If the model has strictly gaussian independent uncertainties then the
        negative log likelihood function will return 0.5*sum(residuals**2),
        which is 1/2*chisq.  Since we are printing normalized chisq, we
        multiply the model nllf by 2/DOF before displaying the value.  This
        is different from the problem nllf function, which includes the
        cost of the cost of the penalty constraints in the total nllf.

        Parameter priors, if any, are treated as independent models
        in the total nllf.  The constraint value is displayed separately.

        **Deprecated**: *norm:bool* and *compact:bool* are ignored.
        """
        chisq_norm, chisq_err = nllf_scale(dof=self.dof, npars=len(self._parameters), norm=norm)
        failing = []
        if nllf is None:
            pparameter, pconstraints, pmodel, failing = self._nllf_components()
            nllf = pmodel + pparameter + pconstraints + (self.penalty_nllf if failing else 0)
            # print(f"{pmodel=} {pparameter=} {pconstraints=} {nllf=} {chisq_norm=} {chisq_err=}")
        if failing:
            # Text is used in a context like f"χ² = {text}". We are not printing the list
            # of failing constraints since parameter names are arbitrarily long.
            text = "NaN [out of bounds]"
        else:
            text = format_uncertainty(nllf * chisq_norm, chisq_err)

        # return f"{text} p={self.getp()} => {pmodel:.15e}"
        return text

    def _nllf_components(self) -> util.Tuple[float, float, float, util.List[str]]:
        try:
            pparameter, bad_priors = self.parameter_nllf()
            pconstraints, bad_constraints = self.constraints_nllf()
            failing = bad_priors + bad_constraints
            # If constraints are failing don't bother to compute nllf.
            # Using pvalue = zero rather than NaN so that handling of penalties is easier.
            pmodel = self.model_nllf() if len(failing) == 0 else 0.0

            if isnan(pparameter):
                # TODO: make sure errors get back to the user
                info = ["Parameter nllf is wrong"]
                info += ["%s %g" % (p, p.nllf()) for p in self._priors]
                logging.error("\n  ".join(info))

            return pparameter, pconstraints, pmodel, failing
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

    @property
    def parameters(self):
        """Return the list of fitted parameters."""
        return self._parameters

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

        # Check priors
        broken = []
        for p in all_parameters:
            if not hasattr(p, "reset_prior"):
                continue
            p.reset_prior()
            limits = p.prior.limits
            value = p.value
            if (limits[0] > value) or (value > limits[1]):
                broken.append(f"{p}={value} is outside {limits}")
            elif not np.isfinite(p.prior.nllf(value)):
                broken.append(f"{p}={value} is outside {p.prior}")

        # Check other constraints.
        broken.extend([f"{c} fails" for c in self.constraints if float(c) > 0])
        # Show broken constraints. Note that warnings.warn() will only show up once,
        # so this acts as a pre-check of the model when running in batch mode on the
        # command line.
        if len(broken) > 0:
            warnings.warn("Unsatisfied constraints: [%s]" % (",\n".join(broken)))

        # TODO: return broken_constraints from reset() rather than setting state
        self.broken_constraints = broken

        # Collect all fitting parameters
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

        # Collect all priors that need to be evaluated
        self._priors = priors(all_parameters)

        # The degrees of freedom is the number of data points minus the number of fit
        # parameters. Gaussian priors on the parameters are treated as a simultaneous
        # fit to the data plus a fit of the parameter to its previously measured value.
        # That is, each prior adds a single data point to the fit without changing the
        # number of free parameters, thus increasing DOF by 1. Again, the target value
        # for a fit with no systematic error should be the number of degrees of freedom.
        # Uniform priors do not modify the degrees of freedom, not even soft-bounded uniform.
        self._dof = self.model_points() + sum(p.prior.dof for p in self._priors) - len(self._parameters)
        if self.dof <= 0:
            warnings.warn(
                f"Need more data points (currently: {self.model_points()}) than fitting parameters ({len(self._parameters)})"
            )

    def model_points(self):
        """Return number of points in all models"""
        return sum(f.numpoints() for f in self.models)

    def model_update(self):
        """Let all models know they need to be recalculated"""
        # The self.models iterator calls update for each model if there are
        # free variable substitutions, so no need to update here.
        if self.freevars:
            return
        for f in self.models:
            getattr(f, "update", lambda: None)()

    def model_nllf(self):
        """Return cost function for all data sets"""
        # print("In model nllf with", self.getp())
        return sum(w**2 * f.nllf() for w, f in zip(self.weights, self.models))

    def constraints_nllf(self) -> util.Tuple[float, util.List[str]]:
        """Return the cost function for all constraints"""
        failing = []
        nllf = 0.0
        # Process the list of inequality constraints
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
            fitness_show_parameters(f, subs=self.freevars.get_model(i))
        print("[overall chisq=%s, nllf=%g]" % (self.chisq_str(), self.nllf()))

    def plot(self, p=None, fignum=1, figfile=None, view=None, model_indices=None):
        # TODO: remove duplicate logic from FitProblem.plot() and api.get_data_plot
        import matplotlib.pyplot as plt

        if p is not None:
            self.setp(p)

        # Don't show the figure calling problem.plot(figfile=base_file), as is
        # done during --export=path. If called as problem.plot(), as from a jupyter
        # notebook, then swe should show the plot.
        show_fig = not figfile

        # Overall chisq
        overall_chisq_str = self.chisq_str()

        for i, f in enumerate(self.models):
            outfile = f"{figfile}-model{i}.png" if figfile else None
            if model_indices is not None and i not in model_indices:
                continue

            backend = "matplotlib" if hasattr(f, "plot") else "plotly" if hasattr(f, "plotly") else None
            # backend = "plotly" if hasattr(f, "plotly") else "matplotlib" if hasattr(f, "plot") else None
            if backend is None:
                continue

            # TODO: duplicated in bumps.webserver.server.api._get_data_plot_mpl()
            if self.num_models > 1:
                chisq_str = fitness_chisq_str(f)
                chisq = f"χ² = {chisq_str}; overall {overall_chisq_str}"
                title = f"Model {i+1}: {f.name}"
            else:
                chisq = f"χ² = {overall_chisq_str}"
                title = f"{f.name}"
            fontsize = 16

            if backend == "plotly":
                fig = f.plotly()
                # TODO: text offset of (x=0.5em, y=0.5ex)
                text_offset = 0.01  # portion of graph axis length
                font = dict(size=16)
                fig.add_annotation(
                    x=text_offset,
                    y=1 + text_offset,
                    xanchor="left",
                    yanchor="bottom",
                    xref="paper",
                    yref="paper",
                    text=title,
                    showarrow=False,
                    font=font,
                )
                fig.add_annotation(
                    x=1 - text_offset,
                    y=1 + text_offset,
                    xanchor="right",
                    yanchor="bottom",
                    xref="paper",
                    yref="paper",
                    text=chisq,
                    showarrow=False,
                    font=font,
                )

                if outfile:
                    # Note: requires "pip install kaleido"
                    # Note: much slower than matplotlib
                    fig.write_image(outfile)
                if show_fig:
                    # Try to guess whether we are in a jupyter notebook before deciding how
                    # to render the plot.
                    # TODO: gather all figures into one tab when rendering to the browser
                    import sys

                    jupyter = "ipykernel" in sys.modules
                    renderer = None if jupyter else "browser"
                    fig.show(renderer)
                continue

            # If not plotly then we must be using matplotlib.
            # Note that during api.export our matplotlib backend is 'agg' so no plot will show.
            fig = plt.figure(i + fignum)
            f.plot(view=view)

            # Make room for model name and chisq on the top of the plot
            # TODO: attach margins to canvas resize_event so that margins are fixed
            h, w = fig.get_size_inches()
            h_ex = h * 72 / fontsize  # (h in * 72 pt/in) / (fontsize pt/ex) = height in ex
            text_offset = 0.5 / h_ex  # 1/2 ex above and below the text
            top = 1 - 2 / h_ex  # leave 2 ex at the top of the figure
            plt.subplots_adjust(top=top)

            # Add model name and chisq
            transform = fig.transFigure
            x, y = text_offset, 1 - text_offset
            ha, va = "left", "top"
            fig.text(x, y, title, transform=transform, va=va, ha=ha, fontsize=fontsize)
            x, y = 1 - text_offset, 1 - text_offset
            ha, va = "right", "top"
            fig.text(x, y, chisq, transform=transform, va=va, ha=ha, fontsize=fontsize)

            # plt.suptitle("Model %d - %s" % (i, f.name))
            # plt.text(0.01, 0.01, "chisq=%s" % fitness_chisq_str(f), transform=plt.gca().transAxes)
            if outfile:
                plt.savefig(outfile, format="png")

    # Note: restore default behaviour of getstate/setstate rather than
    # inheriting from BaseFitProblem
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# TODO: consider adding nllf_scale to FitProblem.
ONE_SIGMA = 0.68268949213708585


def nllf_scale(dof: int, npars: int, norm: bool = True):
    r"""
    Return the scale factor for reporting the problem nllf as an approximate
    normalized chisq, along with an associated "uncertainty".  The uncertainty
    is the amount that chisq must change in order for the fit to be
    significantly better.

    Parameters:
    -----------
    dof : int
        Degrees of freedom (typically n - p where n is data points, p is parameters)
    npars : int
        Number of fitting parameters
    norm : bool, optional
        If True (default), normalize chisq by degrees of freedom

    From Numerical Recipes 15.6: *Confidence Limits on Estimated Model
    Parameters*, the $1-\sigma$ contour in parameter space corresponds
    to $\Delta\chi^2 = \text{invCDF}(1-\sigma,k)$ where
    $1-\sigma \approx 0.6827$ and $k$ is the number of fitting parameters.

    If *norm* is True (default), then we need to normalize chisq by
    the degrees of freedom. This allows us to assess fit quality as the
    average squared error in each data point, which should be around 1.0
    if the model and measurement uncertainties are correct.
    """
    scale = dof if norm else 1
    if scale <= 0 or np.isnan(scale) or np.isinf(scale):
        return 1.0, 0.0
    else:
        npars = max(npars, 1)
        return 2.0 / scale, chi2.ppf(ONE_SIGMA, npars) / scale


def load_problem(path: Path | str, args: list[str] | None = None):
    """
    Load a model file.

    *path* contains the path to the model file. This could be a python script
    or a previously saved problem, serialized as .json, .cloudpickle, .pickle or .dill

    *args* are any additional arguments to the model.  The sys.argv
    variable will be set such that *sys.argv[1:] == model_options*.
    """
    from .webview.server.state_hdf5_backed import SERIALIZER_EXTENSIONS, deserialize_problem_bytes

    path = Path(path)
    table = {f".{ext}": method for method, ext in SERIALIZER_EXTENSIONS.items()}
    method = table.get(path.suffix, "script")
    if method == "script":
        problem = _load_script_from_path(path, args)
    else:
        # export saved data as binary with encoding utf-8
        data = path.read_bytes()
        problem = deserialize_problem_bytes(data, method)
    # TODO: what is problem.path when we are deserializing from a session file?
    problem.path = str(path.resolve())
    if not getattr(problem, "name", None):
        problem.name = path.stem
    if not getattr(problem, "title", None):
        problem.title = path.name

    # Guard against the user changing parameters after defining the problem.
    problem.model_reset()
    return problem


def _load_script_from_path(path: Path | str, args: list[str] | None = None):
    from .util import pushdir
    from . import plugin

    # Change to the target path before loading model so that data files
    # can be given as relative paths in the model file.  Add the directory
    # to the python path (at the end) so that imports work as expected.
    path = Path(path)
    directory, filename = path.parent, path.name
    with pushdir(directory):
        # Try a specialized model loader
        problem = plugin.load_model(filename)
        if problem is None:
            problem = _load_script(filename, options=args)
    # Note: keeping problem.script_path separate from problem.path because
    # the problem.path may be the result of deserializing the model.
    problem.script_path = str(path.resolve())
    problem.script_args = args
    return problem


def _load_script(filename, options=None) -> FitProblem:
    """
    Load a problem definition from a python script file.

    sys.argv is set to ``[file] + options`` within the context of the script.

    The user must define ``problem=FitProblem(...)`` within the script.

    Raises ValueError if the script does not define problem.

    Namespace for imports is `bumps.user`
    """
    import re
    from pathlib import Path
    from hashlib import md5
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    script_path = Path(filename).resolve()
    if options is None:
        options = ()

    # Turn filename into a python identifier
    name = script_path.stem.split(".", 1)[0]
    name = "_".join(name.split())  # convert whitespace
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)  # handle invalid characters
    if name[0].isdigit():  # idnetifiers can't start with a digit
        name = "_" + name

    # Put all user scripts in the bumps.user namespace.
    # They shouldn't conflict because they don't show up
    # in sys.modules.
    package = "bumps.user"

    # Create the module for the script
    fullname = f"{package}.{name}"
    loader = SourceFileLoader(fullname, str(script_path))
    spec = spec_from_loader(fullname, loader)
    module = module_from_spec(spec)

    # Execute the script
    # TODO: Enable relative imports with ScriptFinder
    old_argv = sys.argv
    old_bytecode = sys.dont_write_bytecode
    # meta_path_finder = ScriptFinder(script_path.parent, package)
    try:
        # sys.meta_path.insert(0, meta_path_finder)
        sys.argv = [filename, *options]
        sys.dont_write_bytecode = True  # Suppress .pyc creation
        loader.exec_module(module)
    finally:
        # sys.meta_path.pop(0)
        sys.argv = old_argv
        sys.dont_write_bytecode = old_bytecode

    problem = getattr(module, "problem", None)
    if problem is None:
        raise ValueError(filename + " requires 'problem = FitProblem(...)'")

    # # Capture the source code and any dependent libraries. On deserialize we
    # # will need to preload the libs and stuff them in sys.modules before
    # # calling dill. Be sure to remove them immediately so that the next load
    # # from script will get the latest version (the user may have changed the
    # # support libraries without having changed the script).
    # # Note that we won't be able to rerun the script because we aren't capturing
    # # the original datafiles, but we may want to show it to the user.
    # script = script_path.read_text()
    # context = dict(script=script, libs=meta_path_finder.sources, options=options)

    return problem


# Note: Not currently used
class ScriptFinder:
    """
    sys.meta_path finder allowing relative imports in scripts.

    *script_dir* is the parent directory for the script file.

    *package_name* is the module namespace for the script.

    *sources* contains {fullname: source_code} for all the supporting
    modules in the script directory. Use these to deserialize the
    saved problem.
    """

    def __init__(self, script_dir, package_name):
        from importlib.machinery import ModuleSpec

        self._path = script_dir
        self._package = package_name + "."
        self._parent_spec = ModuleSpec(package_name, None, is_package=True)
        self.sources = {}

    def find_spec(self, fullname, path, target=None):
        from importlib.machinery import SourceFileLoader
        from importlib.util import spec_from_loader

        if fullname == self._package[:-1]:
            # print(f'import looking for {fullname}')
            return self._parent_spec
        if fullname.startswith(self._package):
            # print(f'import looking for {fullname}')
            module_name = fullname.split(".")[-1]
            module_path = self._path / f"{module_name}.py"
            if module_path.exists():
                loader = SourceFileLoader(fullname, str(module_path))
                spec = spec_from_loader(fullname, loader)
                # It is inefficient to load this twice, but the import
                # hook machinery is too complicated to capture the source
                # when it is loaded. Similarly for suppressing the .pyc
                # file creation.
                self.sources[fullname] = module_path.read_text()
                return spec
        return None


def MultiFitProblem(*args, **kwargs) -> FitProblem:
    warnings.warn(DeprecationWarning("use FitProblem directly instead of MultiFitProblem"))
    return FitProblem(*args, **kwargs)


def test_weighting_and_priors():
    class SimpleFitness(Fitness):
        def __init__(self, a=0.0, name="fit"):
            self.a = parameter.Parameter.default(a, name=name + " a")

        def parameters(self):
            return {"a": self.a}

        def numpoints(self):
            return 1

        def residuals(self):
            y, dy = 0, 1  # fit 0 +/- 1 to a constant
            return np.array([(self.a.value - y) / dy])

        def nllf(self):
            return sum(r**2 for r in self.residuals()) / 2

    weights = 2, 3
    M0, M1 = SimpleFitness(4.0, name="M0"), SimpleFitness(5.0, name="M1")
    problem = FitProblem((M0, M1), weights=weights)

    # Need to use problem.models to cycle through models in case FreeVariables is used in problem
    assert (problem.residuals() == np.hstack([w * M.residuals() for w, M in zip(weights, problem.models)])).all()
    assert problem.nllf() == sum(w**2 * M.nllf() for w, M in zip(weights, problem.models))
    assert problem.nllf() == sum(problem.residuals() ** 2) / 2

    # Test priors: constraint on expression
    M0.a.range(-10, 10)  # Set M0.a = 4 to be in bounds
    # TODO: should setting equals() clear the constraints?
    # If we set a constraint then assign an expression the constraint stays on
    # the expression. The other order fails because the expression is considered
    # "fixed" and can't accept constraints.
    M1.a.range(0, 1)  # Set M1.a to be out of bounds
    M1.a.equals(M0.a * 2)  # Set M1.a = 2*M0.a = 4*2 = 8
    problem.model_reset()
    # print(f"{problem._parameters=}, {problem._priors=} {problem._dof=}")
    assert problem._parameters == [M0.a]  # only M0.a is fitted
    assert problem._priors == [M0.a, M1.a]  # both M0.a and M1.a are bounded
    assert problem._dof == 1
    nllf, failing = problem.parameter_nllf()
    assert np.isinf(nllf)
    assert failing == [str(M1.a)]

    M1.a.unlink()
    M1.a.dev(mean=0, std=1)  # Set M1.a to be in bounds but with a cost
    M1.a.equals(M0.a * 2)  # Set M1.a = 2*M0.a = 4*2 = 8
    problem.model_reset()
    # print(f"{problem._parameters=}, {problem._priors=} {problem._dof=}")
    assert problem._parameters == [M0.a]  # only M0.a is fitted
    assert problem._priors == [M0.a, M1.a]  # both M0.a and M1.a are bounded
    # DOF is 2 because we have two data points plus one prior minus one fit parameter
    assert problem._dof == 2
    nllf, failing = problem.parameter_nllf()
    assert nllf == 32  # (8 - 0)**2/(2 * 1)
    assert failing == []


if __name__ == "__main__":
    test_weighting_and_priors()
