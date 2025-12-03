"""

Notes on random numbers
=======================

Uses dream.util.rng as the random number generator.

You can set the seed using::

    dream.util.rng = numpy.random.RandomState(seed)
"""

# TODO: move the rng control to the Dream class rather than a shared global
# This interface doesn't feel right, since one instance of DREAM may
# influence another if they are running within one another.  Putting
# the rng on the dream class may be a better option.

# TODO: Maybe reenable these three exports; they might be used by third parties
# Suppress these imports because I was getting some circular import problems
# from .model import MCMCModel
# from .core import Dream, Model, dream
# from .state import MCMCDraw

# from .initpop import cov_init, lhs_init
# from .model import MCMCModel, Density, LogDensity, Simulation, MVNormal, Mixture
# from .state import MCMCDraw, Draw, dream_load, h5load, h5dump #, load_state save_state
# from .views import plot_all, plot_corrmatrix, plot_corr, plot_traces, plot_logp, plot_acceptance_rate
# from .util import console, draw
# from .stats import var_stats, format_num, format_vars, save_vars, parse_var
