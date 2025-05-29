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
    "new_model",
    "load_model",
    "calc_errors",
    "show_errors",
    "data_view",
    "model_view",
    "migrate_serialized",
]

import importlib.metadata
import logging
import traceback

from typing import Callable, Dict, Generic, List, Optional, Protocol, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from bumps.fitproblem import FitProblem
    import numpy as np
    import matplotlib.pyplot as plt
    import wx

ACTIVE_PLUGIN_NAME: Optional[str] = None


def load_plugin_group(group: str):
    """
    Load all plugins in the specified group.

    This function returns a list of loaded plugins for the specified group.
    """
    eps = importlib.metadata.entry_points().select(group=group)
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
MIGRATION_PLUGINS: Dict[str, Callable[[dict], dict]] = load_plugin_group("bumps.serialization.migration")


"""
Every registered bumps.model.load plugin should be a function that takes
a filename and returns a model object or None if the file does not
contain a model of the appropriate type.
"""
MODEL_LOADER_PLUGINS: Dict[str, Callable[[str], "FitProblem"]] = load_plugin_group("bumps.model.load")


"""
Every registered bumps.model.new plugin should be a function that returns
a new model object.
"""
NEW_MODEL_PLUGINS: Dict[str, Callable[[], "FitProblem"]] = load_plugin_group("bumps.model.new")


"""
Every registered bumps.errplot.matplotlib plugin should be a function that
takes a FitProblem, sample data, and an optional matplotlib Axes,
and displays the model with uncertainty on that Axes object (or the current axes if None).
"""
ERRPLOT_PLUGINS: Dict[str, Callable[["FitProblem", "np.ndarray", Optional["plt.Axes"]], None]] = load_plugin_group(
    "bumps.errplot.matplotlib"
)


"""
Every registered bumps.wx_gui.data_view plugin should be a wx.Panel
that provides a data view for the FitProblem.
"""
DATA_VIEW_PLUGINS: Dict[str, "wx.Panel"] = load_plugin_group("bumps.wx_gui.data_view")


"""
Every registered bumps.wx_gui.model_view plugin should be a wx.Panel
that provides a model view for the FitProblem.
"""
MODEL_VIEW_PLUGINS: Dict[str, "wx.Panel"] = load_plugin_group("bumps.wx_gui.model_view")


"""
Every registered bumps.serialize.save_json plugin should be a function
that takes a FitProblem and a path string, and saves the FitProblem
as a JSON file to that path.
"""
SAVE_JSON_PLUGINS: Dict[str, Callable[["FitProblem", str], None]] = load_plugin_group("bumps.serialize.save_json")


# TODO: refl1d wants to do the following after cli.getopts()
#
#    from refl1d.probe import Probe
#    Probe.view = opts.plot
#
# It also wants to modify the opts so that more plotters are available,
# such as Fresnel.
