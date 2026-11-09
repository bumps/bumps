"""Optional Matplotlib adapters for Bumps sensitivity scans.

This module is deliberately separate from the numerical scanner core.
Install matplotlib only if static plots are required.

The function names and scan data conventions mirror
``sensitivity_plot``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _prepare_y_values(q_values: Any, y_scale: str):
    """Convert profile-statistic values to chi-square or Gaussian-equivalent sigma units."""
    values = np.asarray(q_values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Profile statistic values must be finite.")
    if np.any(values < -1e-8):
        raise ValueError("Profile statistic values must be non-negative.")
    values = np.maximum(values, 0.0)

    if y_scale == "chi2":
        return values, (1.0, 4.0), "Δχ² (1 d.o.f.)", ("Δχ² = 1", "Δχ² = 4")
    if y_scale == "sigma":
        return np.sqrt(values), (1.0, 2.0), "Gaussian-equivalent |z| (σ)", ("1σ", "2σ")
    raise ValueError("y_scale must be either 'chi2' or 'sigma'.")


def _save_figure(fig: Any, path: str | Path | None) -> None:
    """Save a Matplotlib figure when a destination path is provided."""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")


def plot_profile(
    scanner: Any,
    *,
    ax: Any = None,
    show: bool = False,
    title: str | None = None,
    path: str | Path | None = None,
    y_scale: str = "chi2",
):
    """Plot a completed profile scan using chi-square or sigma units.

    Returns the Matplotlib figure and axes as ``(figure, axes)``.

    ``y_scale="chi2"`` displays q directly as Delta chi-square, with reference
    levels 1 and 4. ``y_scale="sigma"`` displays sqrt(q), with reference
    levels 1 sigma and 2 sigma.
    """
    import matplotlib.pyplot as plt

    if not getattr(scanner, "results", None) or scanner.start_stat is None:
        raise RuntimeError("The profile scan must be completed before plotting.")

    values = np.asarray([point[0] for point in scanner.results], dtype=float)
    q_values = np.asarray(
        [point[1] - scanner.start_stat for point in scanner.results],
        dtype=float,
    )
    y_values, thresholds, yaxis_title, threshold_labels = _prepare_y_values(q_values, y_scale)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
    else:
        fig = ax.figure

    ax.scatter(values, y_values, s=18, label="Profile points")
    ax.axhline(0.0, ls=":", lw=1, label="Minimum")
    ax.axhline(thresholds[0], ls="--", lw=1, label=threshold_labels[0])
    ax.axhline(thresholds[1], ls="--", lw=1, label=threshold_labels[1])
    ax.axvline(scanner.best_val, ls="-.", lw=1, label="Best-fit value")

    boundary_suffix = "Δχ² = 1" if y_scale == "chi2" else "1σ"
    for value, label in (
        (scanner.lower_bound, f"Lower {boundary_suffix} boundary"),
        (scanner.upper_bound, f"Upper {boundary_suffix} boundary"),
    ):
        if value is not None:
            ax.axvline(value, ls=":", lw=1, label=label)

    ax.set_xlabel(f"Parameter: {scanner.target_param_name}")
    ax.set_ylabel(yaxis_title)
    ax.set_title(title or f"Bumps profile scan: {scanner.target_param_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    _save_figure(fig, path)
    if show:
        plt.show()
    return fig, ax


def plot_profile_comparison(
    scanners: Iterable[Any],
    *,
    labels=None,
    ax: Any = None,
    show: bool = False,
    title: str = "Comparison of profile scan resolutions",
    path: str | Path | None = None,
    y_scale: str = "chi2",
):
    """Overlay completed profiles using chi-square or sigma units.

    ``y_scale`` accepts ``"chi2"`` (q) or ``"sigma"`` (sqrt(q)).
    """
    import matplotlib.pyplot as plt

    scanners = list(scanners)
    if not scanners:
        raise ValueError("No scanners were provided.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
    else:
        fig = ax.figure

    labels = list(labels) if labels is not None else [f"Scan {index + 1}" for index in range(len(scanners))]
    if len(labels) != len(scanners):
        raise ValueError("The number of labels must match the number of scanners.")

    _, thresholds, yaxis_title, threshold_labels = _prepare_y_values([0.0], y_scale)
    for scanner, label in zip(scanners, labels):
        if not getattr(scanner, "results", None) or scanner.start_stat is None:
            continue
        x_values = np.asarray([point[0] for point in scanner.results], dtype=float)
        q_values = np.asarray(
            [point[1] - scanner.start_stat for point in scanner.results],
            dtype=float,
        )
        y_values, _, _, _ = _prepare_y_values(q_values, y_scale)
        ax.scatter(x_values, y_values, s=16, label=str(label))

    ax.axhline(thresholds[0], ls="--", lw=1, label=threshold_labels[0])
    ax.axhline(thresholds[1], ls=":", lw=1, label=threshold_labels[1])
    ax.set_xlabel(f"Parameter: {scanners[0].target_param_name}")
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    _save_figure(fig, path)
    if show:
        plt.show()
    return fig, ax


def plot_bumps_fit(
    problem: Any,
    fit_result: Any = None,
    *,
    path: str | Path | None = None,
    view=None,
    show: bool = False,
):
    """Use Bumps' native Matplotlib ``FitProblem.plot`` implementation.

    This adapter intentionally delegates model visualization to Bumps. The
    returned object is the Matplotlib figure created or selected by Bumps.
    """
    import matplotlib.pyplot as plt

    plotter = getattr(problem, "plot", None)
    if not callable(plotter):
        raise TypeError("This FitProblem does not provide a callable plot() method.")

    params = getattr(fit_result, "x", fit_result) if fit_result is not None else None
    before = set(plt.get_fignums())
    kwargs = {}
    if params is not None:
        kwargs["p"] = np.asarray(params, dtype=float)
    if view is not None:
        kwargs["view"] = view

    try:
        signature = inspect.signature(plotter)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "fignum" in signature.parameters:
        kwargs["fignum"] = max(before, default=0) + 1

    plotter(**kwargs)
    after = set(plt.get_fignums())
    new_numbers = sorted(after - before)
    fig = plt.figure(new_numbers[-1]) if new_numbers else plt.gcf()

    _save_figure(fig, path)
    if show:
        plt.show()
    return fig


def plot_profile_curves(
    curves,
    *,
    parameter_name: str,
    ax: Any = None,
    title: str = "Profile scan comparison",
    show: bool = False,
    path: str | Path | None = None,
    y_scale: str = "chi2",
):
    """Plot pre-extracted profile curves in chi-square or sigma units.

    ``y_scale`` accepts ``"chi2"`` (q) or ``"sigma"`` (sqrt(q)).
    """
    import matplotlib.pyplot as plt

    curves = list(curves)
    if not curves:
        raise ValueError("No profile curves were provided.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
    else:
        fig = ax.figure

    _, thresholds, yaxis_title, threshold_labels = _prepare_y_values([0.0], y_scale)
    for label, x_values, q_values in curves:
        x = np.asarray(x_values, dtype=float)
        q = np.asarray(q_values, dtype=float)
        if x.shape != q.shape:
            raise ValueError(f"Curve '{label}': x and q have different shapes.")
        y, _, _, _ = _prepare_y_values(q, y_scale)
        order = np.argsort(x)
        ax.scatter(x[order], y[order], s=16, label=str(label))

    ax.axhline(thresholds[0], ls="--", lw=1, label=threshold_labels[0])
    ax.axhline(thresholds[1], ls=":", lw=1, label=threshold_labels[1])
    ax.set_xlabel(f"Parameter: {parameter_name}")
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    _save_figure(fig, path)
    if show:
        plt.show()
    return fig, ax
