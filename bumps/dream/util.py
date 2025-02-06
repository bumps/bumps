"""
Miscellaneous utilities.
"""

__all__ = ["draw", "console"]

import numpy as np
import numpy.random as rng

try:
    from numba import njit
except ImportError:

    def njit(*args, **kw):
        return lambda f: f


@njit(cache=True)
def draw(k, n):
    """
    Select k things from a pool of n without replacement.
    """
    # At k == n/4, an extra 0.15*k draws are needed to get k unique draws
    # TODO: silently returns too few values if k > n
    if k > n / 4:
        result = rng.permutation(n)[:k]
    else:
        result = np.empty(k, np.int64)
        for i in range(k):
            # select an item not already selected
            while True:
                p = rng.randint(n)
                for j in range(i):
                    # if there is a match then break out of the for-loop and
                    # go to the next while iteration, generating a new proposal
                    if j == p:
                        break
                # unusual syntax: if you make it to the end of the for loop
                # then there were no matches, so break out of the while loop
                else:
                    break
            result[i] = p
    return result


def _check_uniform_draw():
    """
    Draws from history should
    """
    import pylab

    k, n = 50, 400
    counts = np.zeros(n * k)
    idx = np.arange(k)
    for _ in range(100000):
        t = draw(k, n)
        counts[k * t + idx] += 1
    pylab.subplot(211)
    pylab.pcolormesh(np.reshape(counts, (n, k)))
    pylab.colorbar()
    pylab.title("drawn number vs. draw position")
    pylab.subplot(212)
    pylab.hist(counts)
    pylab.title("number of draws per (number,position) bin")
    pylab.show()


def console():
    """
    Start the python console with the local variables available.

    console() should be the last thing in the file, after sampling and
    showing the default plots.
    """
    import os
    import sys

    # Hack for eclipse console: can't actually run ipython in the eclipse
    # console and get it to plot, so instead guess whether we are in a
    # console by checking if we are attached to a proper tty through stdin.
    # For eclipse, just show the plots.
    try:
        tty = os.isatty(sys.stdin.fileno())
    except Exception:
        tty = False

    if tty:
        # Display outstanding plots and turn interactive on
        from matplotlib import interactive
        from matplotlib._pylab_helpers import Gcf

        for fig in Gcf.get_all_fig_managers():
            try:  # CRUFT
                fig.show()
            except AttributeError:
                fig.frame.Show()
        interactive(True)

        # Start an ipython shell with the caller's local variables
        import IPython

        symbols = sys._getframe(1).f_locals
        ip = IPython.Shell.IPShell(user_ns=symbols)
        ip.mainloop()
    else:
        # Not a tty; try doing show() anyway
        import pylab

        pylab.show()
