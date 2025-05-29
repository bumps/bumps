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

import importlib.metadata
import logging
import traceback

from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from bumps.fitproblem import FitProblem
    import numpy as np
    import matplotlib.pyplot as plt
    import wx

ACTIVE_PLUGIN_NAME: Optional[str] = None

ENTRY_POINTS = importlib.metadata.entry_points()


def load_plugin_group(group: str):
    """
    Load all plugins in the specified group.

    This function returns a list of loaded plugins for the specified group.
    """
    if getattr(ENTRY_POINTS, "select", None) is None:
        # For importlib.metadata versions < 3.10, use the deprecated method
        eps = ENTRY_POINTS.get(group, [])
    else:
        # For importlib.metadata versions >= 3.10, use the select method
        eps = ENTRY_POINTS.select(group=group)
    outputs = dict()
    for ep in eps:
        try:
            plugin = ep.load()
        except Exception as e:
            # Log the error and continue loading other plugins
            logging.error(f"Failed to load plugin {ep.name}: {e}")
            traceback.print_exc()
        if ep.name in outputs:
            logging.warning(f"Duplicate plugin name found: {ep.name}. Overwriting previous entry.")
        outputs[ep.name] = plugin
    return outputs


"""
Every registered bumps.serialization.migration plugin should be a
function that takes a bumps-serialized (JSON) dictionary and applies
transformations to migrate older versions to the current schema required
for the library, returning the transformed dictionary.
"""
MIGRATION: Dict[str, Callable[[dict], dict]] = load_plugin_group("bumps.serialization.migration")


"""
Every registered bumps.model.load plugin should be a function that takes
a filename and returns a model object or None if the file does not
contain a model of the appropriate type.
"""
MODEL_LOADER: Dict[str, Callable[[str], "FitProblem"]] = load_plugin_group("bumps.model.load")


"""
Every registered bumps.model.new plugin should be a function that returns
a new model object.
"""
NEW_MODEL: Dict[str, Callable[[], "FitProblem"]] = load_plugin_group("bumps.model.new")


"""
Every registered bumps.calc_errors plugin should be a function
that takes a FitProblem and a set of points in parameter space,
and returns an object representing errors for the model at those points.
(that object will be plotted by the show_errors plugin).
"""
CALC_ERRORS: Dict[str, Callable[["FitProblem", "np.ndarray"], Any]] = load_plugin_group("bumps.calc_errors")


"""
Every registered bumps.show_errors.matplotlib plugin should be a function
that takes an object representing errors and an optional matplotlib Axes,
and displays the confidence regions on that Axes object (or the current axes if None).
(The object is the output of calc_errors).
"""
SHOW_ERRORS: Dict[str, Callable[[Any, Optional["plt.Axes"]], None]] = load_plugin_group("bumps.show_errors.matplotlib")


"""
Every registered bumps.wx_gui.data_view plugin should be a wx.Panel
that provides a data view for the FitProblem.
"""
DATA_VIEW: Dict[str, "wx.Panel"] = load_plugin_group("bumps.wx_gui.data_view")


"""
Every registered bumps.wx_gui.model_view plugin should be a wx.Panel
that provides a model view for the FitProblem.
"""
MODEL_VIEW: Dict[str, "wx.Panel"] = load_plugin_group("bumps.wx_gui.model_view")


"""
Every registered bumps.serialize.save_json plugin should be a function
that takes a FitProblem and a path string, and saves the FitProblem
as a JSON file to that path.
"""
SAVE_JSON: Dict[str, Callable[["FitProblem", str], None]] = load_plugin_group("bumps.serialize.save_json")


# TODO: refl1d wants to do the following after cli.getopts()
#
#    from refl1d.probe import Probe
#    Probe.view = opts.plot
#
# It also wants to modify the opts so that more plotters are available,
# such as Fresnel.
