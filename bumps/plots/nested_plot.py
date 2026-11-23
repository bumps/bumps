"""
Progress of a nested sampling run: the evidence, the prior volume remaining
and the effective sample size at each iteration, with a table of the current
values and stopping criteria.
"""

from typing import Optional

import numpy as np

from ..nested_sampling import NestedStats

LINE_COLOR = "#1f77b4"
BAND_COLOR = "rgba(31, 119, 180, 0.2)"
TARGET_COLOR = "gray"
# The ln Z axis opens on this many nats below the current estimate. The early
# iterations, far below it, add almost nothing to the evidence and would
# otherwise flatten the curve. Autoscale shows the whole run.
LOGZ_WINDOW = 10.0
AXIS_STYLE = dict(showline=True, showgrid=False, zeroline=False)


def nested_plot(stats: NestedStats, max_points: Optional[int] = None) -> dict:
    """
    Return a Plotly figure of ln Z ± σ, ln X and ESS against iteration, as a
    dictionary suitable for JSON serialization.

    If there are more than *max_points* iterations then the series are thinned.
    """
    n = len(stats.logz)
    step = n // max_points if max_points is not None and n > max_points else 1
    x = np.arange(1, n + 1)[::step]
    logz, logzerr, logvol, ess = (v[::step] for v in (stats.logz, stats.logzerr, stats.logvol, stats.ess))

    def line(y, yaxis, name, **kw):
        return dict(type="scattergl", x=x, y=y, xaxis="x", yaxis=yaxis, name=name, mode="lines", **kw)

    traces = [
        line(logz + logzerr, "y", "ln Z + σ", line=dict(width=0), showlegend=False, hoverinfo="skip"),
        line(logz - logzerr, "y", "ln Z ± σ", line=dict(width=0), fill="tonexty", fillcolor=BAND_COLOR),
        line(logz, "y", "ln Z", line=dict(color=LINE_COLOR)),
        line(logvol, "y2", "ln X", line=dict(color=LINE_COLOR)),
        line(ess, "y3", "ESS", line=dict(color=LINE_COLOR)),
    ]

    def axis(title, domain, **kw):
        return dict(title={"text": title}, domain=domain, anchor="x", **AXIS_STYLE, **kw)

    top = stats.logz[-1] + max(stats.logzerr[-1], 1.0)

    target = dict(
        type="line",
        xref="x domain",
        x0=0,
        x1=1,
        yref="y3",
        y0=stats.n_effective_target,
        y1=stats.n_effective_target,
        line=dict(color=TARGET_COLOR, dash="dash", width=1),
    )
    layout = dict(
        template="simple_white",
        title=dict(text="Nested Sampling", xanchor="center", x=0.5),
        uirevision=1,
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(title={"text": "Iteration"}, anchor="y3", **AXIS_STYLE),
        yaxis=axis("ln Z", [0.70, 1.0], range=[stats.logz[-1] - LOGZ_WINDOW, top]),
        yaxis2=axis("ln X", [0.35, 0.65]),
        yaxis3=axis("ESS", [0.0, 0.30]),
        shapes=[target],
    )
    return {"data": traces, "layout": layout}


def nested_table(stats: NestedStats) -> list[tuple[str, str]]:
    """
    Return *(label, value)* rows describing the current state of the run.
    """
    phase = {"initial": "Initial run", "batch": f"Batch {stats.batch}", "done": "Done"}[stats.phase]
    rows = [
        ("Phase", phase),
        ("ln Z", f"{stats.logz[-1]:.3f} ± {stats.logzerr[-1]:.3f}"),
        ("ESS", f"{stats.ess[-1]:.0f} / {stats.n_effective_target}"),
        ("ln X", f"{stats.logvol[-1]:.3f}"),
        ("Information H", f"{stats.information:.3f} nats"),
        ("Live points", f"{stats.nlive}"),
        ("Iterations", f"{stats.iteration}"),
        ("Likelihood calls", f"{stats.ncall}"),
    ]
    if stats.delta_logz is not None:
        rows.append(("Remaining Δln Z", f"{stats.delta_logz:.3g} / {stats.dlogz_target:g}"))
    if stats.stop is not None:
        rows += [
            ("Stopping value (stops at ≤ 1)", f"{stats.stop:.3g}"),
            ("· from ESS", f"{stats.stop_post:.3g}"),
            ("· from σ(ln Z)", f"{stats.stop_evid:.3g}"),
        ]
    return rows


def _example_stats(**kw) -> NestedStats:
    n = 5
    values = dict(
        phase="batch",
        batch=2,
        logz=np.linspace(-20, -8, n),
        logzerr=np.full(n, 0.1),
        logvol=-np.arange(n, dtype="d"),
        ess=np.arange(1, n + 1, dtype="d"),
        information=3.0,
        nlive=100,
        iteration=n,
        ncall=50,
        dlogz_target=0.01,
        n_effective_target=1000,
        stop_post=2.0,
        stop_evid=10.0,
        stop=10.0,
    )
    return NestedStats(**{**values, **kw})


def test_nested_plot():
    fig = nested_plot(_example_stats(), max_points=2)
    assert [trace["name"] for trace in fig["data"]][2:] == ["ln Z", "ln X", "ESS"]
    assert all(len(trace["x"]) == 3 for trace in fig["data"])
    assert fig["layout"]["yaxis"]["range"][0] == -8 - LOGZ_WINDOW


def test_nested_table():
    batch = dict(nested_table(_example_stats()))
    assert batch["Phase"] == "Batch 2"
    assert batch["ln Z"] == "-8.000 ± 0.100"
    assert batch["Stopping value (stops at ≤ 1)"] == "10"
    assert "Remaining Δln Z" not in batch

    initial = dict(
        nested_table(_example_stats(phase="initial", delta_logz=0.5, stop_post=None, stop_evid=None, stop=None))
    )
    assert initial["Remaining Δln Z"] == "0.5 / 0.01"
    assert "Stopping value (stops at ≤ 1)" not in initial
