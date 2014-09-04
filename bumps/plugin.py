"""
Bumps plugin architecture.

With sophisticated models, developers need to be able to provide tools
such as model builders and data viewers.

Some of these will be tools for the GUI, such as views.  Others will be
tools to display results.

This file defines the interface that can be defined by your own application
so that it interacts with models of your type.  Define your own model
package with a module plugin.py.

Create a main program which looks like::


    if __name__ == "__main__":
        import multiprocessing
        multiprocessing.freeze_support()

        import bumps.cli
        import mypackage.plugin
        bumps.cli.install_plugin(mypackage.plugin)
        bumps.cli.main()

You should be able to use this as a driver program for your application.

Note: the plugin architecture is likely to change radically as more models
are added to the system, particularly so that we can accommodate simultaneous
fitting of data taken using different experimental techniques.  For now, only
only one plugin at a time is supported.
"""

__all__ = [
    'new_model',
    'load_model',
    'calc_errors',
    'show_errors',
    'data_view',
    'model_view',
]

# TODO: refl1d wants to do the following after cli.getopts()
#
#    from refl1d.probe import Probe
#    Probe.view = opts.plot
#
# It also wants to modify the opts so that more plotters are available,
# such as Fresnel.


def new_model():
    """
    Return a new empty model or None.

    Called in response to >File >New from the GUI.  Creates a new empty
    model.  Also triggered if GUI is started without a model.
    """
    return None


def load_model(filename):
    """
    Return a model stored within a file.

    This routine is for specialized model descriptions not defined by script.

    If the filename does not contain a model of the appropriate type (e.g.,
    because the extension is incorrect), then return None.

    No need to load pickles or script models.  These will be attempted if
    load_model returns None.
    """
    return None


def calc_errors(problem, sample):
    """
    Gather data needed to display uncertainty in the model and the data.

    Returns an object to be passed later to :func:`show_errors`.
    """
    return None


def show_errors(errs):
    """
    Display the model with uncertainty on the current figure.

    *errs* is the data returned from calc_errs.
    """
    pass


def data_view():
    """
    Panel factory for the data tab in the GUI.

    If your model has an adequate show() function this should not be
    necessary.
    """
    from .gui.data_view import DataView
    return DataView


def model_view():
    """
    Panel factory for the model tab in the GUI.

    Return None if not present.
    """
    return None
