"""Profile-scan uncertainty analysis for a Bumps FitProblem.

The scanner fixes one Bumps fit parameter at a time and re-optimizes all
other varying parameters.  It uses the Bumps objective function ``nllf`` and
reports the profile statistic

    q(theta) = 2 * (nllf_profile(theta) - nllf_profile_min)

For a regular one-parameter likelihood-ratio problem, q=1 and q=4 are the
usual approximate 1-sigma and 2-sigma thresholds.

Important statistical note
--------------------------
Bumps' ``FitProblem.nllf()`` is not generally the data likelihood alone.  In
the Bumps implementation supplied with this project it includes parameter
prior terms and constraint penalties.  Therefore this scanner profiles the
full Bumps objective.  With informative/non-uniform priors, q=1 and q=4 must
not automatically be interpreted as frequentist profile-likelihood
confidence limits.  See the class docstring and the accompanying usage notes.

Dependencies: numpy and scipy only. Plotting lives in an optional companion module.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence, Any

import numpy as np
from scipy.optimize import minimize


class ProfileLiklehoodScanner:
    """Profile one varying parameter in a Bumps ``FitProblem``.

    Parameters
    ----------
    problem:
        A Bumps FitProblem whose parameters currently contain the fitted
        solution, or a problem to which ``best_fit_result`` will be applied.
    target_param_name:
        Exact label from ``problem.labels()`` / parameter ``name``.
    min_bound, max_bound:
        Finite scan limits. They must lie within the target parameter's Bumps
        prior bounds.
    optimizer_method:
        SciPy optimizer used to re-fit nuisance parameters (default: L-BFGS-B).
    optimizer_options:
        Options forwarded to ``scipy.optimize.minimize``.
    target_param_index:
        Optional zero-based parameter index. Use this to disambiguate repeated
        parameter names in multi-model fits. If supplied, the name must still
        match the label at that index.
    best_fit_result:
        Optional Bumps fit result (an object with ``.x``) or parameter vector.
        When supplied, its parameter vector is loaded into ``problem``.
    require_optimizer_success:
        Raise when SciPy reports an unsuccessful nuisance minimization. Set to
        False only for diagnostic work; unsuccessful points are recorded and
        warned about, not silently treated as fully reliable.

    Notes
    -----
    ``FitProblem.setp`` in the supplied Bumps source already calls
    ``model_update``.  The scanner therefore deliberately does not call
    ``model_update`` a second time.
    """

    def __init__(
        self,
        problem: Any,
        target_param_name: str,
        min_bound: float,
        max_bound: float,
        optimizer_method: str = "L-BFGS-B",
        optimizer_options: Optional[dict] = None,
        *,
        target_param_index: Optional[int] = None,
        best_fit_result: Optional[Any] = None,
        require_optimizer_success: bool = True,
    ) -> None:
        self.problem = problem
        self.target_param_name = str(target_param_name)
        self.min_bound = float(min_bound)
        self.max_bound = float(max_bound)
        self.optimizer_method = optimizer_method
        self.optimizer_options = dict(
            optimizer_options if optimizer_options is not None else {"maxiter": 2000, "ftol": 1e-10}
        )
        self.require_optimizer_success = bool(require_optimizer_success)

        if not (np.isfinite(self.min_bound) and np.isfinite(self.max_bound)):
            raise ValueError("min_bound and max_bound must be finite.")
        if not self.min_bound < self.max_bound:
            raise ValueError("min_bound must be smaller than max_bound.")

        if best_fit_result is not None:
            candidate = getattr(best_fit_result, "x", best_fit_result)
            candidate = np.asarray(candidate, dtype=float)
            if candidate.ndim != 1 or candidate.size == 0:
                raise ValueError(
                    "best_fit_result must contain a one-dimensional parameter vector or an object with a one-dimensional .x attribute."
                )
            self._set_problem_params(candidate)

        self.best_params = self._read_current_params()
        self.parameter_names = self._get_parameter_names()

        matches = [i for i, name in enumerate(self.parameter_names) if name == self.target_param_name]
        if target_param_index is None:
            if len(matches) != 1:
                available = ", ".join(f"{i}: {name}" for i, name in enumerate(self.parameter_names))
                if len(matches) > 1:
                    reason = (
                        f"The parameter name '{self.target_param_name}' occurs more than once. "
                        "Please provide target_param_index."
                    )
                else:
                    reason = f"Parameter '{self.target_param_name}' was not found."
                raise ValueError(f"{reason} Available parameters: {available}")
            self.target_index = matches[0]
        else:
            if not isinstance(target_param_index, (int, np.integer)):
                raise TypeError("target_param_index must be an integer.")
            if not 0 <= int(target_param_index) < self.best_params.size:
                raise IndexError("target_param_index is outside the parameter vector.")
            self.target_index = int(target_param_index)
            actual_name = self.parameter_names[self.target_index]
            if actual_name != self.target_param_name:
                raise ValueError(
                    f"target_param_name='{self.target_param_name}' does not match "
                    f"parameter {self.target_index} ('{actual_name}')."
                )

        self.lower_bounds, self.upper_bounds = self._get_bounds()
        self.best_val = float(self.best_params[self.target_index])

        if not self.lower_bounds[self.target_index] <= self.best_val <= self.upper_bounds[self.target_index]:
            raise ValueError("The current best-fit value is outside the Bumps prior bounds.")
        if (
            self.min_bound < self.lower_bounds[self.target_index]
            or self.max_bound > self.upper_bounds[self.target_index]
        ):
            raise ValueError(
                "The scan limits are outside the Bumps prior bounds: "
                f"Scan [{self.min_bound:g}, {self.max_bound:g}], "
                f"Bumps [{self.lower_bounds[self.target_index]:g}, "
                f"{self.upper_bounds[self.target_index]:g}]."
            )
        if not self.min_bound <= self.best_val <= self.max_bound:
            raise ValueError(
                f"Best-fit value {self.best_val:g} is outside the scan limits [{self.min_bound:g}, {self.max_bound:g}]."
            )

        self.nuisance_indices = np.array(
            [i for i in range(self.best_params.size) if i != self.target_index],
            dtype=int,
        )

        # Public attributes storing the scan results and uncertainty limits.
        self.results: list[tuple[float, float]] = []  # (parameter value, 2*nllf)
        self.lower_bound: Optional[float] = None
        self.upper_bound: Optional[float] = None
        self.lower_2sigma: Optional[float] = None
        self.upper_2sigma: Optional[float] = None
        self.start_nllf: Optional[float] = None
        self.start_stat: Optional[float] = None
        self._best_profile_params = self.best_params.copy()
        self.optimizer_diagnostics: list[dict[str, Any]] = []
        self.direction_diagnostics: dict[str, str] = {}

    def _read_current_params(self) -> np.ndarray:
        try:
            params = np.asarray(self.problem.getp(), dtype=float)
        except Exception as exc:
            raise RuntimeError("Could not read problem.getp().") from exc
        if params.ndim != 1 or params.size == 0 or not np.all(np.isfinite(params)):
            raise ValueError("problem.getp() must return a finite, non-empty one-dimensional vector.")
        return params.copy()

    def _set_problem_params(self, params: Sequence[float]) -> None:
        """Set parameters through Bumps; setp() already triggers model_update()."""
        vector = np.asarray(params, dtype=float)
        if vector.ndim != 1:
            raise ValueError("The parameter vector must be one-dimensional.")
        expected = len(getattr(self.problem, "getp")())
        if vector.size != expected:
            raise ValueError(f"Parameter vector has length {vector.size}; expected {expected}.")
        setter = getattr(self.problem, "setp", None)
        if not callable(setter):
            raise TypeError("The supplied object does not provide a callable setp() method.")
        status = setter(vector.copy())
        # Some Bumps versions return None; an explicit False can signal rejection.
        if status is False:
            raise RuntimeError("Bumps rejected the parameter vector passed to setp().")

    def _get_parameter_names(self) -> list[str]:
        """Return labels in exactly the order used by FitProblem.getp()."""
        labels = None
        label_method = getattr(self.problem, "labels", None)
        if callable(label_method):
            try:
                labels = [str(item) for item in label_method()]
            except Exception:
                labels = None
        if labels is not None and len(labels) == self.best_params.size:
            return labels

        try:
            params = list(self.problem.parameters)
            names = [str(parameter.name) for parameter in params]
            if len(names) == self.best_params.size:
                return names
        except Exception:
            pass

        raise RuntimeError(
            "Parameter names cannot be reliably mapped to the order returned by problem.getp(). "
            "Please check FitProblem.labels() or problem.parameters."
        )

    def _get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Read bounds; the FitProblem implementation supplied uses shape (2, N)."""
        getter = getattr(self.problem, "bounds", None)
        if not callable(getter):
            raise TypeError("The supplied object does not provide a bounds() method.")
        try:
            bounds = np.asarray(getter(), dtype=float)
        except Exception as exc:
            raise RuntimeError("Could not read problem.bounds().") from exc

        n = self.best_params.size
        if bounds.shape == (2, n):
            lower, upper = bounds[0].copy(), bounds[1].copy()
        elif bounds.shape == (n, 2):
            lower, upper = bounds[:, 0].copy(), bounds[:, 1].copy()
        else:
            raise ValueError(f"problem.bounds() has shape {bounds.shape}; expected (2, {n}) or ({n}, 2).")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("problem.bounds() contains invalid parameter bounds.")
        return lower, upper

    def _nllf(self, params: Sequence[float]) -> float:
        self._set_problem_params(params)
        value = float(self.problem.nllf())
        return value if np.isfinite(value) else np.inf

    def _profile_at(
        self, target_value: float, start_params: np.ndarray
    ) -> tuple[float, np.ndarray, Optional[dict[str, Any]]]:
        """Fix target parameter and minimize nllf over all nuisance parameters."""
        fixed = np.asarray(start_params, dtype=float).copy()
        fixed[self.target_index] = float(target_value)

        if self.nuisance_indices.size == 0:
            value = self._nllf(fixed)
            if not np.isfinite(value):
                raise RuntimeError(f"nllf is invalid at {self.target_param_name}={target_value:g}.")
            return 2.0 * value, fixed, None

        x0 = fixed[self.nuisance_indices].copy()
        lo = self.lower_bounds[self.nuisance_indices]
        hi = self.upper_bounds[self.nuisance_indices]

        x0 = np.maximum(x0, lo)
        x0 = np.minimum(x0, hi)
        nuisance_bounds = list(zip(lo.tolist(), hi.tolist()))

        def objective(x: np.ndarray) -> float:
            trial = fixed.copy()
            trial[self.nuisance_indices] = x
            return self._nllf(trial)

        try:
            fit = minimize(
                objective,
                x0,
                method=self.optimizer_method,
                bounds=nuisance_bounds,
                options=self.optimizer_options.copy(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"SciPy optimization failed at {self.target_param_name}={target_value:g}: {exc}"
            ) from exc

        if not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)):
            raise RuntimeError(f"No finite solution at {self.target_param_name}={target_value:g}: {fit.message}")

        diagnostic = {
            "target_value": float(target_value),
            "success": bool(fit.success),
            "status": int(getattr(fit, "status", -1)),
            "message": str(fit.message),
            "nit": int(getattr(fit, "nit", -1)),
            "nfev": int(getattr(fit, "nfev", -1)),
        }
        if not fit.success:
            message = (
                f"Nuisance-parameter optimization at {self.target_param_name}={target_value:g} "
                f"reported failure: {fit.message}"
            )
            if self.require_optimizer_success:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)

        optimized = fixed.copy()
        optimized[self.nuisance_indices] = np.asarray(fit.x, dtype=float)
        optimized[self.target_index] = float(target_value)
        final_nllf = self._nllf(optimized)
        if not np.isfinite(final_nllf):
            raise RuntimeError(f"Invalid nllf value at {self.target_param_name}={target_value:g}.")
        diagnostic["nllf"] = float(final_nllf)
        self.optimizer_diagnostics.append(diagnostic)
        return 2.0 * final_nllf, optimized, diagnostic

    def run_scan(
        self,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        max_steps: int = 2000,
        max_step_fraction: float = 0.0005,
        initial_step_fraction: float = 1e-5,
        threshold: float = 4.0,
    ) -> list[tuple[float, float]]:
        """Scan both sides of the best value and re-fit nuisance parameters.

        ``progress_callback(value, q)`` is called after each successful scan
        point. The returned ``results`` preserve the original format
        ``[(parameter_value, 2*nllf_profile), ...]``.
        """
        if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        if not (0 < initial_step_fraction <= max_step_fraction):
            raise ValueError("The condition 0 < initial_step_fraction <= max_step_fraction must hold.")
        if threshold <= 0 or not np.isfinite(threshold):
            raise ValueError("threshold must be a positive finite number.")

        param_range = self.max_bound - self.min_bound
        max_step = param_range * max_step_fraction
        min_step = max(
            param_range * 1e-12,
            np.finfo(float).eps * max(1.0, abs(self.best_val)) * 10.0,
        )
        step0 = max(param_range * initial_step_fraction, min_step)

        original_params = self._read_current_params()
        raw: list[tuple[float, float]] = []
        self.results = []
        self.lower_bound = None
        self.upper_bound = None
        self.lower_2sigma = None
        self.upper_2sigma = None
        self.start_nllf = None
        self.start_stat = None
        self.optimizer_diagnostics = []
        self.direction_diagnostics = {}

        try:
            base_stat, base_params, _ = self._profile_at(self.best_val, original_params)
            self.start_nllf = 0.5 * base_stat
            self.start_stat = float(base_stat)
            self._best_profile_params = base_params.copy()
            raw.append((self.best_val, base_stat))

            # A meaningful improvement at another target value means that the
            # supplied fit was not the minimum required for a likelihood-ratio scan.
            for direction in (1.0, -1.0):
                label = "right" if direction > 0 else "left"
                step = step0
                current = self.best_val
                warm_start = base_params.copy()
                previous_q = 0.0
                reached_limit = False
                reached_threshold = False
                steps_done = 0

                for _ in range(int(max_steps)):
                    candidate = current + direction * step
                    if candidate < self.min_bound or candidate > self.max_bound:
                        candidate = min(max(candidate, self.min_bound), self.max_bound)
                    if np.isclose(candidate, current, rtol=0.0, atol=min_step):
                        reached_limit = True
                        break

                    stat, optimized, _ = self._profile_at(candidate, warm_start)
                    q = float(stat - base_stat)
                    if q < -1e-6:
                        raise RuntimeError(
                            f"The scan found at {self.target_param_name}={candidate:g} "
                            f"a lower Bumps nllf value than at the supplied best-fit value "
                            f"(Delta q={q:.4g}). Please re-optimize the fit first "
                            "and pass the resulting fit solution to the scanner."
                        )
                    if q < 0.0:
                        q = 0.0
                        stat = base_stat

                    raw.append((float(candidate), float(stat)))
                    steps_done += 1
                    if progress_callback is not None:
                        progress_callback(float(candidate), q)

                    current = float(candidate)
                    warm_start = optimized
                    delta_q = max(0.0, q - previous_q)
                    previous_q = max(previous_q, q)

                    if q >= threshold:
                        reached_threshold = True
                        break
                    if current == self.min_bound or current == self.max_bound:
                        reached_limit = True
                        break

                    if delta_q > 0.05:
                        step = max(step / 2.0, min_step)
                    elif delta_q < 0.005:
                        step = min(step * 1.2, max_step)
                    else:
                        step = min(step * 1.05, max_step)

                    if step <= min_step and delta_q < 1e-12:
                        warnings.warn(
                            f"Minimum step size reached on the {label} side.",
                            RuntimeWarning,
                        )
                        break

                self.direction_diagnostics[label] = (
                    "q threshold reached"
                    if reached_threshold
                    else "scan boundary reached"
                    if reached_limit
                    else f"max_steps reached/scan stopped ({steps_done} steps)"
                )
                if not reached_threshold and not reached_limit:
                    warnings.warn(
                        f"On the {label} side, neither q={threshold:g} nor the scan boundary was reached.",
                        RuntimeWarning,
                    )
        finally:
            # Restore the exact state present when run_scan() was called.
            self._set_problem_params(original_params)

        # Sort and deduplicate scan points.
        deduplicated: dict[float, float] = {}
        for x, stat in raw:
            deduplicated[float(x)] = float(stat)
        self.results = sorted(deduplicated.items(), key=lambda item: item[0])
        return self.results

    @staticmethod
    def _find_crossing(
        points: list[tuple[float, float]],
        center_x: float,
        center_stat: float,
        delta_q: float,
        side: str,
    ) -> Optional[float]:
        target_stat = center_stat + delta_q
        points = sorted(points, key=lambda item: item[0], reverse=(side == "left"))
        previous = (center_x, center_stat)
        for point in points:
            x1, y1 = previous
            x2, y2 = point
            if min(y1, y2) <= target_stat <= max(y1, y2):
                if np.isclose(y1, y2):
                    return None
                return float(x1 + (target_stat - y1) * (x2 - x1) / (y2 - y1))
            previous = point
        return None

    def extract_uncertainties(self) -> tuple[Optional[float], Optional[float]]:
        """Interpolate q=1 and q=4 crossings; do not extrapolate beyond scan."""
        if not self.results or self.start_stat is None:
            raise RuntimeError("Run run_scan() first.")

        left = [(x, y) for x, y in self.results if x < self.best_val]
        right = [(x, y) for x, y in self.results if x > self.best_val]
        self.lower_bound = self._find_crossing(left, self.best_val, self.start_stat, 1.0, "left")
        self.upper_bound = self._find_crossing(right, self.best_val, self.start_stat, 1.0, "right")
        self.lower_2sigma = self._find_crossing(left, self.best_val, self.start_stat, 4.0, "left")
        self.upper_2sigma = self._find_crossing(right, self.best_val, self.start_stat, 4.0, "right")

        if self.lower_bound is None:
            warnings.warn("The left q=1 crossing was not reached within the scan range.", RuntimeWarning)
        if self.upper_bound is None:
            warnings.warn("The right q=1 crossing was not reached within the scan range.", RuntimeWarning)
        return self.lower_bound, self.upper_bound

    def get_summary(self) -> dict[str, Any]:

        if self.start_stat is None:
            raise RuntimeError("Run run_scan() first.")
        return {
            "parameter": self.target_param_name,
            "parameter_index": self.target_index,
            "best_value": self.best_val,
            "nllf_profile_min": self.start_nllf,
            "lower_1sigma_q1": self.lower_bound,
            "upper_1sigma_q1": self.upper_bound,
            "lower_2sigma_q4": self.lower_2sigma,
            "upper_2sigma_q4": self.upper_2sigma,
            "statistic": "q = 2 * (profiled Bumps nllf - minimum profiled Bumps nllf)",
            "interpretation": (
                "Profile likelihood only under suitable likelihood/prior assumptions; "
                "Bumps nllf includes prior and constraint terms."
            ),
            "scan_points": [
                {"value": float(x), "twice_nllf": float(stat), "q": float(stat - self.start_stat)}
                for x, stat in self.results
            ],
            "optimizer_diagnostics": list(self.optimizer_diagnostics),
            "direction_diagnostics": dict(self.direction_diagnostics),
        }

    def print_summary(self) -> None:
        if not self.results or self.start_stat is None:
            print("No scan results are available.")
            return
        print("\n" + "=" * 66)
        print(f" PROFILE SCAN: {self.target_param_name} (Index {self.target_index})")
        print("=" * 66)
        print(f"Scan points:                        {len(self.results)}")
        print(f"Best-fit value:                           {self.best_val:.8g}")
        if self.lower_bound is not None and self.upper_bound is not None:
            print(f"q=1 interval:                      [{self.lower_bound:.8g}, {self.upper_bound:.8g}]")
            print(
                f"Asymmetric uncertainty:         "
                f"-{self.best_val - self.lower_bound:.5g} / "
                f"+{self.upper_bound - self.best_val:.5g}"
            )
        else:
            print("q=1 interval: at least one boundary was not reached.")
        if self.lower_2sigma is not None or self.upper_2sigma is not None:
            print(f"q=4 boundaries:                         {self.lower_2sigma} / {self.upper_2sigma}")
        print("Reference: q = 2 * Delta(Bumps nllf); q=1 and q=4 are approximations.")
        print("Note: Bumps nllf may include prior and constraint contributions.")
        for side, status in self.direction_diagnostics.items():
            print(f"Scan {side:5s}:                         {status}")
        print("=" * 66 + "\n")

    def execute_workflow(self, progress_callback=None):
        """Run the profile scan and extract limits without plotting dependencies."""
        self.run_scan(progress_callback=progress_callback)
        self.extract_uncertainties()
        self.print_summary()
        return self.get_summary()
