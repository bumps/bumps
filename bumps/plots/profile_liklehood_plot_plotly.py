"""Optional Plotly adapters for Bumps sensitivity scans.

This module is deliberately separate from the numerical scanner core.
Install plotly only if interactive plots are required.
"""

from __future__ import annotations

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


def _save_html(fig: Any, path: str | Path | None) -> None:
    """Save a self-contained interactive Plotly figure as an HTML file."""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def plot_profile(
    scanner: Any,
    *,
    fig: Any = None,
    show: bool = False,
    title: str | None = None,
    path: str | Path | None = None,
    y_scale: str = "chi2",
):
    """Plot a completed profile scan using chi-square or sigma units.

    Returns a Plotly Figure. Pass an existing Plotly Figure with ``fig`` to
    add the profile to an existing interactive plot.

    ``y_scale="chi2"`` displays q directly as Delta chi-square, with reference
    levels 1 and 4. ``y_scale="sigma"`` displays sqrt(q), with reference
    levels 1 sigma and 2 sigma.
    """
    import plotly.graph_objects as go

    if not getattr(scanner, "results", None) or scanner.start_stat is None:
        raise RuntimeError("The profile scan must be completed before plotting.")

    values = np.asarray([point[0] for point in scanner.results], dtype=float)
    q_values = np.asarray(
        [point[1] - scanner.start_stat for point in scanner.results],
        dtype=float,
    )
    y_values, thresholds, yaxis_title, threshold_labels = _prepare_y_values(q_values, y_scale)

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=values,
            y=y_values,
            mode="markers",
            name="Profile points",
            marker={"size": 6},
            hovertemplate=f"Parameter: %{{x}}<br>{'Δχ²' if y_scale == 'chi2' else 'σ'}: %{{y:.6g}}<extra></extra>",
        )
    )

    fig.add_hline(y=0.0, line_dash="dot", annotation_text="Minimum")
    fig.add_hline(y=thresholds[0], line_dash="dash", annotation_text=threshold_labels[0])
    fig.add_hline(y=thresholds[1], line_dash="dash", annotation_text=threshold_labels[1])
    fig.add_vline(
        x=scanner.best_val,
        line_dash="dashdot",
        annotation_text="Best-fit value",
    )

    boundary_suffix = "Δχ² = 1" if y_scale == "chi2" else "1σ"
    for value, label in (
        (scanner.lower_bound, f"Lower {boundary_suffix} boundary"),
        (scanner.upper_bound, f"Upper {boundary_suffix} boundary"),
    ):
        if value is not None:
            fig.add_vline(
                x=value,
                line_dash="dot",
                annotation_text=label,
            )

    fig.update_layout(
        title=title or f"Bumps profile scan: {scanner.target_param_name}",
        xaxis_title=f"Parameter: {scanner.target_param_name}",
        yaxis_title=yaxis_title,
        template="plotly_white",
        width=900,
        height=550,
        hovermode="closest",
    )

    _save_html(fig, path)
    if show:
        fig.show()
    return fig


def plot_profile_comparison(
    scanners: Iterable[Any],
    *,
    labels=None,
    fig: Any = None,
    show: bool = False,
    title: str = "Comparison of profile scan resolutions",
    path: str | Path | None = None,
    y_scale: str = "chi2",
):
    """Overlay completed profiles using chi-square or sigma units.

    Returns a Plotly Figure. Incomplete scanners are skipped, matching the
    behavior of the Matplotlib adapter.

    ``y_scale="chi2"`` displays q directly; ``y_scale="sigma"`` displays
    sqrt(q) as a Gaussian-equivalent absolute z-score.
    """
    import plotly.graph_objects as go

    scanners = list(scanners)
    if not scanners:
        raise ValueError("No scanners were provided.")

    labels = list(labels) if labels is not None else [f"Scan {index + 1}" for index in range(len(scanners))]
    if len(labels) != len(scanners):
        raise ValueError("The number of labels must match the number of scanners.")

    if fig is None:
        fig = go.Figure()

    for scanner, label in zip(scanners, labels):
        if not getattr(scanner, "results", None) or scanner.start_stat is None:
            continue

        x_values = np.asarray([point[0] for point in scanner.results], dtype=float)
        q_values = np.asarray(
            [point[1] - scanner.start_stat for point in scanner.results],
            dtype=float,
        )
        y_values, thresholds, yaxis_title, threshold_labels = _prepare_y_values(q_values, y_scale)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=str(label),
                marker={"size": 5},
                hovertemplate=(
                    f"Parameter: %{{x}}<br>{'Δχ²' if y_scale == 'chi2' else 'σ'}: %{{y:.6g}}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    _, thresholds, yaxis_title, threshold_labels = _prepare_y_values([0.0], y_scale)
    fig.add_hline(y=thresholds[0], line_dash="dash", annotation_text=threshold_labels[0])
    fig.add_hline(y=thresholds[1], line_dash="dot", annotation_text=threshold_labels[1])
    fig.update_layout(
        title=title,
        xaxis_title=f"Parameter: {scanners[0].target_param_name}",
        yaxis_title=yaxis_title,
        template="plotly_white",
        width=900,
        height=550,
        hovermode="closest",
    )

    _save_html(fig, path)
    if show:
        fig.show()
    return fig


def plot_profile_curves(
    curves,
    *,
    parameter_name: str,
    fig: Any = None,
    title: str = "Profile scan comparison",
    show: bool = False,
    path: str | Path | None = None,
    y_scale: str = "chi2",
):
    """Plot pre-extracted profile curves in chi-square or sigma units.

    ``y_scale`` accepts ``"chi2"`` (q) or ``"sigma"`` (sqrt(q)).
    """
    import plotly.graph_objects as go

    curves = list(curves)
    if not curves:
        raise ValueError("No profile curves were provided.")

    if fig is None:
        fig = go.Figure()

    for label, x_values, q_values in curves:
        x = np.asarray(x_values, dtype=float)
        q = np.asarray(q_values, dtype=float)
        if x.shape != q.shape:
            raise ValueError(f"Curve '{label}': x and q have different shapes.")
        y, thresholds, yaxis_title, threshold_labels = _prepare_y_values(q, y_scale)

        order = np.argsort(x)
        fig.add_trace(
            go.Scatter(
                x=x[order],
                y=y[order],
                mode="markers",
                name=str(label),
                marker={"size": 5},
                hovertemplate=(
                    f"Parameter: %{{x}}<br>{'Δχ²' if y_scale == 'chi2' else 'σ'}: %{{y:.6g}}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    _, thresholds, yaxis_title, threshold_labels = _prepare_y_values([0.0], y_scale)
    fig.add_hline(y=thresholds[0], line_dash="dash", annotation_text=threshold_labels[0])
    fig.add_hline(y=thresholds[1], line_dash="dot", annotation_text=threshold_labels[1])
    fig.update_layout(
        title=title,
        xaxis_title=f"Parameter: {parameter_name}",
        yaxis_title=yaxis_title,
        template="plotly_white",
        width=900,
        height=550,
        hovermode="closest",
    )

    _save_html(fig, path)
    if show:
        fig.show()
    return fig
