"""
Interactive console widget support.

Defines NumpyConsole class.

TODO: Fix cut/paste for multiline commands
TODO: Trigger change notification when numpy array has changed
"""

import wx, wx.py


def shapestr(v):
    """Return shape string for numeric variables suitable for printing"""
    try:
        shape = v.shape
    except AttributeError:
        return "scalar"
    else:
        return "array " + "x".join([str(i) for i in shape])


class NumpyConsole(wx.py.shell.ShellFrame):
    """
    NumpyConsole defines an interactive console which is aware of all the
    numerical variables in the local name space.  When variables are added
    or removed, it signals that the set of variables has changed.

    This is intended to be used as an embedded console in an interactive
    application for which the user can define and manipulate numerical
    types and automatically show a list of variables, e.g., available for
    plotting.

    If you subclass and replace self.init_code, be sure to do so _before_ calling
    the superclass __init__.
    """

    # code to define the initial namespace
    init_code = """
from pylab import *
"""
    introText = """
Welcome to the numpy console!

Example:

x = linspace(0,1,300)
y = sin(2*pi*x*3)
vars()
plot(x,y)
"""

    def __init__(self, *args, **kwargs):
        # Start interpreter and monitor statement execution
        msg = kwargs.pop("introText", self.introText)
        wx.py.shell.ShellFrame.__init__(self, *args, **kwargs)

        # Print welcome message.
        # TODO: replace this when (if?) ShellFrame allows introText=msg
        print(msg, file=self.shell)
        self.shell.prompt()

        # Initialize the interpreter namespace with useful commands
        self.shell.interp.runcode(compile(self.init_code, "__main__", "exec"))

        # steal draw_if_interactive
        import pylab
        from matplotlib._pylab_helpers import Gcf
        from matplotlib import pyplot

        self._dirty = set()

        def draw_if_interactive():
            # print "calling draw_if_interactive with",Gcf.get_active()
            self._dirty.add(Gcf.get_active())

        pyplot.draw_if_interactive = draw_if_interactive

        # add vars command to the interpreter
        self.shell.interp.locals["vars"] = self._print_vars

        # ignore the variables defined by numpy
        self.ignore = set(self.shell.interp.locals.keys())
        self._existing = {}  # No new variables recorded yet

        # remember which variables are current so we can detect changes
        wx.py.dispatcher.connect(receiver=self._onPush, signal="Interpreter.push")

    def filter(self, key, value):
        """
        Return True if var should be listed in the available variables.
        """
        return key not in self.ignore

    # Dictionary interface
    def items(self):
        """
        Return the list of key,value pairs for all locals not ignored.
        """
        locals = self.shell.interp.locals
        for k, v in locals.items():
            if self.filter(k, v):
                yield k, v

    def update(self, *args, **kw):
        """
        Update a set of variables from a dictionary.
        """
        self.shell.interp.locals.update(*args, **kw)
        self._existing.update(*args, **kw)

    def __setitem__(self, var, val):
        """
        Define or replace a variable in the interpreter.
        """
        self.shell.interp.locals[var] = val
        self._existing[var] = val

    def __getitem__(self, var):
        """
        Retrieve a variable from the interpreter.
        """
        return self.shell.interp.locals[var]

    def __delitem__(self, var):
        """
        Delete a variable from the interpreter.
        """
        del self.shell.interp.locals[var]
        try:
            del self._existing[var]
        except KeyError:
            pass

    # Stream interface
    def write(self, msg):
        """
        Support 'print >>console, blah' for putting output on console.

        TODO: Maybe redirect stdout to console if console is open?
        """
        self.shell.write(self, msg)

    # ==== Internal messages ====
    def OnChanged(self, added=[], changed=[], removed=[]):
        """
        Override this method to perform your changed operation.

        Note that we cannot detect changes within a variable without considerable
        effort: we would need to keep a deep copy of the original value, and
        use a deep comparison to see if it has changed.
        """
        for var in added:
            print("added", var, file=self.shell)
        for var in changed:
            print("changed", var, file=self.shell)
        for var in removed:
            print("deleted", var, file=self.shell)
        print("override the OnChanged message to update your application state", file=self.shell)

    def _print_vars(self):
        """
        Print the available numeric variables and their shapes.

        This is a command available to the user as vars().
        """
        locals = self.shell.interp.locals
        for k, v in self.items():
            print(k, shapestr(v), file=self.shell)

    def _onPush(self, **kw):
        """On command execution, detect if variable list has changed."""
        # Note: checking for modify is too hard ... build it into the types?
        # print >>self.shell, "checking for add/delete..."
        # Update graphs if changed
        if self._dirty:
            import pylab
            from matplotlib._pylab_helpers import Gcf

            # print "figs",Gcf.figs
            # print "dirty",self._dirty
            for fig in self._dirty:
                # print fig, Gcf.figs.values(),fig in Gcf.figs.values()
                if fig and fig in Gcf.figs.values():
                    # print "drawing"
                    fig.canvas.draw()
            pylab.show()
            self._dirty.clear()

        items = dict(list(self.items()))
        oldkeys = set(self._existing.keys())
        newkeys = set(items.keys())
        added = newkeys - oldkeys
        removed = oldkeys - newkeys
        changed = set(k for k in (oldkeys & newkeys) if items[k] is not self._existing[k])
        if added or changed or removed:
            self.OnChanged(added=added, changed=changed, removed=removed)
        self._existing = items


def demo():
    """Example use of the console."""
    import numpy as np

    app = wx.App(redirect=False)
    ignored = {"f": lambda x: 3 + x}
    console = NumpyConsole(locals=ignored)
    console.update({"x": np.array([[42, 15], [-10, 12]]), "z": 42.0})
    console.Show(True)
    app.MainLoop()


if __name__ == "__main__":
    demo()
