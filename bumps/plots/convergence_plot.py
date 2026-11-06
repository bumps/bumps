from typing import Optional
import numpy as np

TRIM_COLOR = "orange"
BURN_COLOR = "red"
DLOGZ_COLOR = "purple"


def convergence_plot(
    convergence: np.ndarray,
    dof: float,
    cutoff: float = 0.25,
    trim_index: Optional[int] = None,
    burn_index: Optional[int] = None,
    max_points: Optional[int] = None,
    dlogz: Optional[np.ndarray] = None,
) -> dict:
    """
    Generate a convergence plot from the convergence data.

    Args:
        convergence: A 2D numpy array where the first column contains the best values
        and the remaining columns contain the population values.

        dof: Degrees of freedom for normalization.

        cutoff: Fraction of the leading steps to drop from the plot.

        trim_index: Step at which the usable samples start, or None for no marker.

        burn_index: Step at which burn-in ends, or None for no marker.

        max_points: Maximum number of points to draw per trace.

        dlogz: Running estimate of the log evidence still to be collected,
        one value per step of *convergence*, or None to omit it. Steps where
        the estimate is undefined are NaN and are drawn as gaps.

    Returns:
        A dictionary suitable for JSON serialization containing the plot data and layout.
    """

    # Normalize the population data
    normalized_pop = 2 * convergence / dof
    best, pop = normalized_pop[:, 0], normalized_pop[:, 1:]
    ni, npop = pop.shape
    x = np.arange(1, ni + 1)
    tail = int(cutoff * ni)
    nout = ni - tail

    step = nout // max_points if (max_points is not None and nout > max_points) else 1
    x_out = x[tail::step]
    best_out = best[tail::step]
    pop_out = pop[tail::step, :]
    # dlogz starts at step 1 like the rest, so plotly pairs it with x_out correctly
    # even if the two histories differ in length by a step. JSON has no NaN, so
    # the undefined steps are sent as null, which plotly draws as a gap.
    dlogz_out = (
        None if dlogz is None else [v if np.isfinite(v) else None for v in np.asarray(dlogz, "d")[tail::step]]
    )

    # print(f"Convergence plot: {ni} steps, {npop} population, {nout} output, step={step}")
    traces = []
    layout = {}
    hovertemplate = "(%{{x}}, %{{y}})<br>{label}<extra></extra>"
    if npop == 5:
        if (trim_index is not None and trim_index > tail) or (burn_index is not None and burn_index > tail):
            layout["shapes"] = []
            layout["annotations"] = []
        # fig['data'].append(dict(type="scattergl", x=x, y=pop[tail:,4].tolist(), name="95%", mode="lines", line=dict(color="lightblue", width=1), showlegend=True, hovertemplate=hovertemplate.format(label="95%")))
        if trim_index is not None and trim_index > tail:
            layout["shapes"].append(
                {
                    "line": {"color": TRIM_COLOR, "dash": "dash", "width": 2},
                    "type": "line",
                    "x0": trim_index,
                    "x1": trim_index,
                    "xref": "x",
                    "y0": 0,
                    "y1": 0.95,
                    "yref": "y domain",
                },
            )
            layout["annotations"].append(
                {
                    "text": "←trim<br>use→",
                    "x": trim_index,
                    "y": 0.95,
                    "xref": "x",
                    "xanchor": "left",
                    "yref": "y domain",
                    "showarrow": False,
                    "font": {"color": TRIM_COLOR, "size": 12},
                },
            )
        if burn_index is not None and burn_index > tail:
            layout["shapes"].append(
                {
                    "line": {"color": BURN_COLOR, "dash": "dash", "width": 2},
                    "type": "line",
                    "x0": burn_index,
                    "x1": burn_index,
                    "xref": "x",
                    "y0": 0,
                    "y1": 1,
                    "yref": "y domain",
                }
            )
            layout["annotations"].append(
                {
                    "text": "←burn<br>samples→",
                    "x": burn_index,
                    "y": 1.0,
                    "xref": "x",
                    "xanchor": "left",
                    "yref": "y domain",
                    "showarrow": False,
                    "font": {"color": BURN_COLOR, "size": 12},
                }
            )

        traces.append(
            dict(
                type="scattergl",
                x=x_out,
                y=pop_out[:, 3],
                mode="lines",
                line=dict(color="lightgreen", width=0),
                showlegend=False,
                hovertemplate=hovertemplate.format(label="80%"),
            )
        )
        traces.append(
            dict(
                type="scattergl",
                x=x_out,
                y=pop_out[:, 1],
                name="20% to 80% range",
                fill="tonexty",
                mode="lines",
                line=dict(color="lightgreen", width=0),
                hovertemplate=hovertemplate.format(label="20%"),
            )
        )
        # # Including worst makes for very ugly plots when there are outlier chains.
        # # Even without outliers, the chisq distribution has heavy tails, making it
        # # difficult to see the details of the 60% band in the center.
        # traces.append(
        #     dict(
        #         type="scattergl",
        #         x=x_out,
        #         y=pop_out[:, -1],
        #         name="population worst",
        #         mode="lines",
        #         line=dict(color="green", dash="dot"),
        #     )
        # )
        traces.append(
            dict(
                type="scattergl",
                x=x_out,
                y=pop_out[:, 2],
                name="population median",
                mode="lines",
                line=dict(color="green"),
                opacity=0.5,
            )
        )
        traces.append(
            dict(
                type="scattergl",
                x=x_out,
                y=pop_out[:, 0],
                name="population best",
                mode="lines",
                line=dict(color="green", dash="dot"),
            )
        )

    traces.append(
        dict(
            type="scattergl",
            x=x_out,
            y=best_out,
            name="best",
            line=dict(color="red", width=1),
            mode="lines",
        )
    )
    if dlogz is not None:
        traces.append(
            dict(
                type="scattergl",
                x=x_out,
                y=dlogz_out,
                name="remaining Δlog <i>Z</i>",
                line=dict(color=DLOGZ_COLOR, width=1),
                mode="lines",
                yaxis="y2",
                hovertemplate=hovertemplate.format(label="remaining Δlog Z"),
            )
        )
    layout.update(
        dict(
            template="simple_white",
            legend=dict(x=1, y=1, xanchor="right", yanchor="top"),
        )
    )
    layout.update(dict(title=dict(text="Convergence", xanchor="center", x=0.5)))
    layout.update(dict(uirevision=1))
    layout.update(
        dict(
            xaxis=dict(
                title={"text": "Steps"},
                showline=True,
                showgrid=False,
                zeroline=False,
            )
        )
    )
    layout.update(
        dict(
            yaxis=dict(
                title={"text": "Normalized <i>\u03c7</i><sup>2</sup>"}, showline=True, showgrid=False, zeroline=False
            )
        )
    )
    if dlogz is not None:
        # Δlog Z does not share units with the normalized chisq, so it needs its own
        # scale. Tint the axis to match the trace so it is clear which is which. It
        # falls by orders of magnitude towards the stopping threshold, so use log.
        layout.update(
            dict(
                yaxis2=dict(
                    title={"text": "remaining Δlog <i>Z</i>", "font": {"color": DLOGZ_COLOR}},
                    type="log",
                    tickfont={"color": DLOGZ_COLOR},
                    linecolor=DLOGZ_COLOR,
                    overlaying="y",
                    side="right",
                    showline=True,
                    showgrid=False,
                    zeroline=False,
                )
            )
        )
    return {
        "data": traces,
        "layout": layout,
    }
