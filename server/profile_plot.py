# Below taken from the GUI profile plot methods in Refl1d - modified to be stand alone
# This is to allow for the layer labels to be easily added to the SLD plots

# Originally written for use in a jupyter notebook
# Adapted to act as a prototype for the profile plotting in the new GUI for refl1d

from refl1d.view.config import thickness_color, pick_radius
from numpy import inf
from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
blend_xy = transforms.blended_transform_factory


# === Sample information ===
class FindLayers:
    def __init__(self, experiment, axes=None, x_offset=0):

        self.experiment = experiment
        self.axes = axes
        self.x_offset = x_offset
        if (axes is not None) and isinstance(axes, plt.Axes):
            self.xcoords = blend_xy(axes.transData, axes.transAxes)
        self._show_labels = True
        self._show_boundaries = True

        self._find_layer_boundaries()

    def find(self, z):
        return self.experiment.sample.find(z - self.x_offset)

    def _find_layer_boundaries(self):
        offset = self.x_offset
        boundary = [-inf, offset]
        if hasattr(self.experiment, 'sample'):
            for L in self.experiment.sample[1:-1]:
                dx = L.thickness.value
                offset += dx
                boundary.append(offset)
        boundary.append(inf)
        self.boundary = np.asarray(boundary)

    def sample_labels(self):
        return [L.name for L in self.experiment.sample]

    def sample_layer(self, n):
        return self.experiment.sample[n]

    def label_offsets(self):
        z = self.boundary[1:-1]
        left = -20
        middle = [(a + b) / 2. for a, b in zip(z[:-1], z[1:])]
        right = z[-1] + 150
        return [left] + middle + [right]

    def reset_markers_mpl(self):
        """
        Reset all markers, for matplotlib plots
        """
        # self.clear_markers()
        if not isinstance(self.axes, plt.Axes):
            raise ValueError("reset_markers_mpl can only be used with type: axes=matplotlib.axes.Axes")

        ax = self.axes

        # Add bars
        style = dict(linewidth=1, linestyle='-',
                     color=thickness_color, alpha=0.75,
                     pickradius=pick_radius,
                     visible=self._show_boundaries,
                     )
        self.markers = [ax.axvline(x=z, **style)
                        for z in self.boundary[1:-1]]

        # fittable = [self.profile.sample_layer(idx).thickness.fittable
        #             for idx,_ in enumerate(self.markers)]
        # fittable[0] = False # First interface is not fittable
        # for f,m in zip(fittable,self.markers):
        #     if not f: m.set(linestyle='--', linewidth=1.25)
        # self.connect_markers(m for f,m in zip(fittable,self.markers) if f)

        # Add labels
        offsets = self.label_offsets()
        labels = self.sample_labels()
        style = dict(transform=self.axes.get_xaxis_transform(),
                     ha='left', va='bottom',
                     rotation=45, fontsize=16,  # 'medium',
                     visible=self._show_labels,
                     )
        self.textmark = [ax.text(z, 1, s, **style)
                         for z, s in zip(offsets, labels)]

    def reset_markers_plotly(self):
        """
        Reset all markers, for plotly plots
        """
        # self.clear_markers()
        if not isinstance(self.axes, go.Figure):
            raise ValueError("reset_markers_plotly can only be used with type: axes=plotly.graph_objs.Figure")

        fig = self.axes

        # Add bars
        fittable = [self.experiment.sample[idx].thickness.fittable
                    for idx, _ in enumerate(self.boundary[1:-1])]
        fittable[0] = False  # First interface is not fittable

        for f, z_pos in zip(fittable, self.boundary[1:-1]):
            if not f:
                line_dict = dict(dash="dash")
            else:
                line_dict = dict(dash="solid")

            fig.add_vline(x=z_pos, opacity=0.75,
                          line=line_dict,
                          )

        # for f,m in zip(fittable,self.markers):
        #     if not f: m.set(linestyle='--', linewidth=1.25)
        # self.connect_markers(m for f,m in zip(fittable,self.markers) if f)

        # Add labels
        offsets = self.label_offsets()
        labels = self.sample_labels()
        for label_pos, label in zip(offsets, labels):
            fig.add_annotation(text=label,
                               textangle=-30,
                               x=label_pos,
                               yref="paper",
                               y=1.0,
                               yanchor="bottom",
                               showarrow=False
                               )


def determine_problem_fitness(problem, active_model):
    if hasattr(problem, 'active_model'):
        problem_fitness = problem.set_active_model(active_model)
        problem_fitness = problem.active_model.fitness
    else:
        problem_fitness = problem.fitness

    return problem_fitness


def generate_best_profile(model):
    if hasattr(model, "active_model"):
        model = model.active_model

    if model.ismagnetic:
        best = model.magnetic_smooth_profile()
    else:
        best = model.smooth_profile()
    return best
# ============================================================================= #
# Plotting script below
# ============================================================================= #


def plot_sld_profile(model):
    """
    Basic plot script to plot up SLD profiles using Paul's layer finding code
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    theta_axis = ax.twinx()
    # ax.legend(fontsize=16, framealpha=0.5, ncol=3)

    if model.ismagnetic:
        z_best, rho_best, irho_best, rhoM_best, thetaM_best = generate_best_profile(model)
        ax.plot(z_best, rhoM_best, label=r"$\rho_{M}$ mSLD", color="b")
        ax.plot(z_best, thetaM_best, label=r"Magnetic Angle $\theta_{M} \circ$", color="orange")
        ax.set_ylabel(r"$\rho, \rho_{i}, \rho_{M} \left(10^{-6}\AA^{-2}\right)$", fontsize=18)
    else:
        z_best, rho_best, irho_best = generate_best_profile(model)
        ax.set_ylabel(r"$\rho, \rho_{i} \left(10^{-6}\AA^{-2}\right)$", fontsize=18)

    ax.plot(z_best, rho_best, label=r"$\rho$ SLD", color="k")
    ax.plot(z_best, irho_best, label=r"$\rho_{i}$ Im SLD", color="g")

    ax.legend(fontsize=16, framealpha=0.5, ncol=3)
    ax.grid(True)
    ax.set_xlabel(r"z $\left(\AA\right)$", fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)

    layer_markers = FindLayers(model, axes=ax)
    layer_markers.reset_markers_mpl()

    return fig, ax


def plot_sld_profile_plotly(model):
    fig = go.Figure()
    if model.ismagnetic:
        z_best, rho_best, irho_best, rhoM_best, thetaM_best = generate_best_profile(model)

        fig.add_scatter(x=z_best, y=thetaM_best,
                          name="θ<sub>M</sub>", yaxis="y2",
                          hovertemplate='(%{x}, %{y})<br>'
                                        'Theta M'
                                        '<extra></extra>',
                          line={"color": "gold"})
        # TODO: need to make axis scaling for thetaM dependent on if thetaM exceeds 0-360
        fig.update_layout(yaxis2={
            'title': {'text': 'Magnetic Angle θ<sub>M</sub> / °'},
            'type': 'linear',
            'autorange': False,
            'range': [0, 360],
            'anchor': 'x',
            'overlaying': 'y',
            'side': 'right',
            'ticks': "inside",
            # 'ticklen': 20,
        })

        fig.add_scatter(x=z_best, y=rhoM_best, name="ρ<sub>M</sub>",
                          hovertemplate='(%{x}, %{y})<br>'
                                        'M SLD'
                                        '<extra></extra>',
                          line={"color": "blue"})
        yaxis_title = 'SLD: ρ, ρ<sub>i</sub>, ρ<sub>M</sub> / 10<sup>-6</sup> Å<sup>-2</sup>'

    else:
        z_best, rho_best, irho_best = generate_best_profile(model)
        yaxis_title = 'SLD: ρ, ρ<sub>i</sub> / 10<sup>-6</sup> Å<sup>-2</sup>'

    fig.add_scatter(x=z_best, y=irho_best, name="ρ<sub>i</sub>",
                      hovertemplate='(%{x}, %{y})<br>'
                                    'Im SLD'
                                    '<extra></extra>',
                      line={"color": "green"})
    fig.add_scatter(x=z_best, y=rho_best, name="ρ",
                      hovertemplate='(%{x}, %{y})<br>'
                                    'SLD'
                                    '<extra></extra>',
                      line={"color": "black"})

    fig.update_layout(uirevision=1, plot_bgcolor="white")

    fig.update_layout(xaxis={
        'title': {'text': 'depth (Å)'},
        'type': 'linear',
        'autorange': True,
        # 'gridcolor': "Grey",
        'ticks': "inside",
        # 'ticklen': 20,
        'showline': True,
        'linewidth': 1,
        'linecolor': 'black',
        'mirror': "ticks",
        'side': 'bottom'
    })

    fig.update_layout(yaxis={
        'title': {'text': yaxis_title},
        'exponentformat': 'e',
        'showexponent': 'all',
        'type': 'linear',
        'autorange': True,
        # 'gridcolor': "Grey",
        'ticks': "inside",
        # 'ticklen': 20,
        'showline': True,
        'linewidth': 1,
        'linecolor': 'black',
        'mirror': True
    })

    fig.update_layout(margin={
        "l": 75,
        "r": 50,
        "t": 50,
        "b": 75,
        "pad": 4
    })

    fig.update_layout(legend={
        "x": -0.1,
        "bgcolor": "rgba(255,215,0,0.15)",
        "traceorder": "reversed"
    })

    marker_positions = FindLayers(model, axes=fig)
    marker_positions.reset_markers_plotly()

    # fig.show(renderer='firefox')
    return fig

# def plot_contours(model, title, savepath=None, nsamples=1000, show_contours=True, show_mag=True, savetitle=None,
#                   store=None, ultranest=False, dream=False, align=0):
#     if show_contours:
#         # data, columns = generate_contour_data(model)
#         if ultranest:
#             points_array = model.post_samples
#             sub_array = points_array[np.sort(np.random.randint(points_array.shape[0], size=nsamples)), :]
#             errors = calc_errors(model.problem, sub_array)
#             data, columns = generate_profile_data(errors)
#         # if this comment is removed the above code will not overwrite the previous reference to the errors object
#         # and it will act as a bad memory leak - using more ram everytime the function is called
#         if dream:
#             state, points = load_dream_state_notebook(problem=model.problem, store=store)
#             errors = calc_errors(model.problem, points[-nsamples:-1])
#             data, columns = generate_profile_data(errors, align)
#
#     # get best profile manually as we are not using DREAM state
#     if show_mag:
#         z_best, rho_best, irho_best, rhoM_best, thetaM_best = generate_best_profile(model)
#     else:
#         z_best, rho_best, irho_best = generate_best_profile(model)
#
#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#
#     # rho
#     if show_contours:
#         z, best, sig2lo, sig2hi, siglo, sighi = data[0]
#         ax.plot(z, best, label=r"$\rho$ SLD", color="orange")
#         ax.fill_between(z, siglo, sighi, alpha=0.5, color="orange", label=r"$\rho$ SLD $1\sigma$")
#         ax.fill_between(z, sig2lo, sig2hi, alpha=0.25, color="orange", label=r"$\rho$ SLD $2\sigma$")
#     else:
#         ax.plot(z_best, rho_best, label=r"$\rho$ SLD", color="orange")
#
#     # irho
#     # ax.plot(z_best, irho_best, label=r"$\rho_{i}$ Im SLD")
#     # if show_contours:
#     #     z, best, sig2lo, sig2hi, siglo, sighi = data[1]
#     #     ax.fill_between(z, siglo, sighi, alpha=0.5)
#     #     ax.fill_between(z, sig2lo, sig2hi, alpha=0.25)
#     if show_mag:
#         if show_contours:
#             z, best, sig2lo, sig2hi, siglo, sighi = data[2]
#             ax.plot(z, best, label=r"$\rho_{M}$ mSLD", color="b")
#             ax.fill_between(z, siglo, sighi, alpha=0.5, color="b", label=r"$\rho_{M}$ mSLD $1\sigma$")
#             ax.fill_between(z, sig2lo, sig2hi, alpha=0.25, color="b", label=r"$\rho_{M}$ mSLD $2\sigma$")
#         else:
#             ax.plot(z_best, rhoM_best, label=r"$\rho_{M}$ mSLD", color="b")
#
#     # ax.legend(fontsize=14, loc='upper right', framealpha=0.5, ncol=2)
#     ax.legend(fontsize=16, framealpha=0.5, ncol=3)
#
#     ax.grid(True)
#     ax.set_xlabel(r"z $\left(\AA\right)$", fontsize=18)
#     ax.set_ylabel(r"$\rho, \rho_{i} \left(10^{-6}\AA^{-2}\right)$", fontsize=18)
#     if show_mag:
#         ax.set_ylabel(r"$\rho, \rho_{M} \left(10^{-6}\AA^{-2}\right)$", fontsize=18)
#     ax.tick_params(axis='y', labelsize=16)
#     ax.tick_params(axis='x', labelsize=16)
#
#     experiment = model.determine_problem_fitness()
#
#     layer_markers = FindLayers(experiment, axes=ax)
#     layer_markers.reset_markers()
#     fig.suptitle(title)
#     # if np.max(rho_best) > 75:
#     #     top = np.max(rho_best)*1.15
#     # else:
#     #     top =75
#     # ax.set_ylim(top=top)
#     if savepath:
#         if savetitle:
#             title = savetitle
#         plt.savefig(f"{savepath}/{title}.png")
