"""
Build trace plots (plotly)
"""

__all__ = ['plot_trace']

import numpy as np

def plot_trace(x: list | np.ndarray, ys: list | np.ndarray, label='', alpha=0.4):

    import plotly.graph_objs as go

    traces = []
    for y in ys:
        traces.append(go.Scatter(x=x,
                                y=y,
                                mode='lines',
                                showlegend=False,
                                opacity=alpha))

    fig = go.Figure(data=traces)

    fig.update_layout(
        template = 'plotly_white',
        xaxis_title=dict(text='Generation number'),
        yaxis_title=dict(text=label),
        yaxis=dict(exponentformat='e'),
        showlegend=False
    )

    return fig