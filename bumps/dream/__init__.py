"""

Notes on random numbers
=======================

Uses dream.util.rng as the random number generator.

You can set the seed using::

    dream.util.rng = numpy.random.RandomState(seed)

This interface doesn't feel right, since one instance of DREAM may
influence another if they are running within one another.  Putting
the rng on the dream class may be a better option.
"""

#from .core import dream
from .initpop import *  # cov_init, lhs_init
from .model import *    #
from .state import *    # load_state, save_state
from .views import *    # plotting routines
from .core import Dream
from .util import console
