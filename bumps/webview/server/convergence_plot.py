from typing import Optional
import numpy as np


def convergence_plot(convergence: np.ndarray, dof: float, cutoff: float = 0.25, burn_index: Optional[int] = None):
    """
    Generate a convergence plot from the convergence data.

    Parameters:
    - convergence: A 2D numpy array where the first column contains the best values
      and the remaining columns contain the population values.
    - dof: Degrees of freedom for normalization.

    Returns:
    - A dictionary suitable for JSON serialization containing the plot data and layout.
    """

    # Normalize the population data
    normalized_pop = 2 * convergence / dof
    best, pop = normalized_pop[:, 0], normalized_pop[:, 1:]

    ni, npop = pop.shape
    x = np.arange(1, ni + 1)
    tail = int(cutoff * ni)
    traces = []
    layout = {}
    hovertemplate = "(%{{x}}, %{{y}})<br>{label}<extra></extra>"
    if npop == 5:
        # fig['data'].append(dict(type="scattergl", x=x, y=pop[tail:,4].tolist(), name="95%", mode="lines", line=dict(color="lightblue", width=1), showlegend=True, hovertemplate=hovertemplate.format(label="95%")))
        if burn_index is not None and burn_index > tail:
            layout["shapes"] = [
                {
                    "line": {"color": "orange", "dash": "dash", "width": 2},
                    "type": "line",
                    "x0": burn_index,
                    "x1": burn_index,
                    "xref": "x",
                    "y0": 0,
                    "y1": 1,
                    "yref": "y domain",
                },
            ]
            layout["annotations"] = [
                {
                    "text": "burn",
                    "x": burn_index,
                    "y": 0.95,
                    "xref": "x",
                    "xanchor": "left",
                    "yref": "y domain",
                    "showarrow": False,
                    "font": {"color": "orange", "size": 12},
                },
            ]

        traces.append(
            dict(
                type="scattergl",
                x=x[tail:],
                y=pop[tail:, 3],
                mode="lines",
                line=dict(color="lightgreen", width=0),
                showlegend=False,
                hovertemplate=hovertemplate.format(label="80%"),
            )
        )
        traces.append(
            dict(
                type="scattergl",
                x=x[tail:],
                y=pop[tail:, 1],
                name="20% to 80% range",
                fill="tonexty",
                mode="lines",
                line=dict(color="lightgreen", width=0),
                hovertemplate=hovertemplate.format(label="20%"),
            )
        )
        traces.append(
            dict(
                type="scattergl",
                x=x[tail:],
                y=pop[tail:, 2],
                name="population median",
                mode="lines",
                line=dict(color="green"),
                opacity=0.5,
            )
        )
        traces.append(
            dict(
                type="scattergl",
                x=x[tail:],
                y=pop[tail:, 0],
                name="population best",
                mode="lines",
                line=dict(color="green", dash="dot"),
            )
        )

    traces.append(
        dict(
            type="scattergl",
            x=x[tail:],
            y=best[tail:],
            name="best",
            line=dict(color="red", width=1),
            mode="lines",
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
        dict(yaxis=dict(title={"text": "<i>\u03c7</i><sup>2</sup>"}, showline=True, showgrid=False, zeroline=False))
    )
    return {
        "data": traces,
        "layout": layout,
    }
