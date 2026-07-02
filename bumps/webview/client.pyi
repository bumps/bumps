class BumpsClient:
    """Proxy for the bumps server api with methods for each api call.

*url* is the url of the running bumps server."""
    def __init__(url: str) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    async def apply_parameters(pathlist: typing.List[str], filename: str) -> Any:
    async def connect() -> None:
        """Connect to the client."""
    async def disconnect() -> None:
        """Disconnect from client."""
    async def export_results(export_path: typing.Union[str, typing.List[str]] = "") -> Any:
    async def get_chisq(problem: typing.Optional[bumps.fitproblem.FitProblem] = None, nllf: Any = None) -> str:
    async def get_convergence_plot(cutoff: float = 0.25, portion: typing.Optional[float] = None, max_points: typing.Optional[int] = 10000) -> Any:
        """Get the convergence plot for the current fit state.
If the fit state is not available, return None.
If the convergence is not available, return None.

:param cutoff: The cutoff value for the convergence plot
                (fraction of points below this value are not shown)
:param max_points: The maximum number of points to plot
                    (thinning applied if too many points)
:return: A JSON-serializable dictionary containing the convergence plot data."""
    async def get_correlation_plot(sort: bool = True, max_rows: int = 8, nbins: int = 50, vars: Any = None, timestamp: str = "") -> Any:
    async def get_custom_plot(model_index: int, plot_title: str, n_samples: int = 1) -> Any:
    async def get_custom_plot_info() -> Any:
    async def get_data_plot(model_indices: typing.Optional[typing.List[int]] = None) -> Any:
    async def get_dirlisting(pathlist: typing.Optional[typing.List[str]] = None) -> Any:
    async def get_fit_fields() -> Any:
    async def get_fitter_defaults() -> Any:
    async def get_history() -> Any:
    async def get_model() -> Any:
    async def get_model_names() -> Any:
    async def get_model_uncertainty_plot() -> Any:
    async def get_parameter_labels() -> Any:
    async def get_parameter_trace_plot(var: int) -> Any:
    async def get_parameters(only_fittable: bool = False) -> Any:
    async def get_serializer() -> Any:
    async def get_session() -> Any:
    async def get_shared_setting(setting: str) -> Any:
    async def get_topic_messages(topic: typing.Optional[typing.Literal['log']] = None, max_num: Any = None) -> typing.List[typing.Dict]:
    async def get_uncertainty_plot(timestamp: str = "") -> Any:
    async def load_problem_file(pathlist: typing.List[str], filename: str, autosave_previous: bool = True, args: typing.List[str] = None) -> Any:
        """Load the problem from json or from a script file.

*pathlist* is a list of folder components and *filename* is the script file in that folder.
These are joined together as "Path(*pathlist, filename)" to build the complete path. If
path is already a Path to the file, use *load_problem_file([path.parent], path.name, ...)*

If *autosave_previous* then store the current problem state in the session file before
loading the new problem (default=True).

*args* are any additional arguments to the script file. This will be available in the script
as *sys.argv[1:]*."""
    async def load_session(pathlist: typing.List[str], filename: str, read_only: bool = False) -> Any:
    async def publish(topic: str, message: typing.Any = None) -> Any:
    async def reload_history_item(name: str) -> Any:
    async def remove_history_item(name: str) -> Any:
    async def save_parameters(pathlist: typing.List[str], filename: str, overwrite: bool = False) -> Any:
    async def save_problem_file(pathlist: typing.Optional[typing.List[str]] = None, filename: typing.Optional[str] = None, overwrite: bool = False) -> Any:
        """Export current problem to a file.

*pathlist* and *file*"""
    async def save_session() -> Any:
    async def save_session_copy(pathlist: typing.List[str], filename: str) -> Any:
    async def save_to_history(label: str, keep: bool = False) -> str:
    async def set_fit_options(fitter_id: str, options: typing.Dict[str, typing.Any]) -> Any:
    async def set_keep_history(name: str, keep: bool) -> Any:
    async def set_parameter(parameter_id: str, property: typing.Literal['value01', 'value', 'min', 'max'], value: typing.Union[float, str, bool]) -> Any:
    async def set_serialized_problem(serialized: Any, new_model: bool = False, name: typing.Optional[str] = None, method: str = "dataclass") -> Any:
        """Set the fit problem from a saved problem state.

*serialized* is the serialized fit problem. *method* is the method used for serialization.

If *new_model* is True, then save the model to history with tag "Loaded model". (default=False)

*name* is an optional override for the model name.

For example::

    await set_serialized_problem(api.state.problem.fitProblem, method=api.state.problem.serializer)"""
    async def set_session_output_file(filepath: typing.Union[str, pathlib._local.Path, NoneType] = None) -> Any:
        """Set the session output file to be used for saving results, and enable autosave.
If `filepath` is None, the session output file is cleared and autosave is disabled."""
    async def set_shared_setting(setting: str, value: typing.Any) -> Any:
    async def set_trim_portion(portion: float) -> Any:
        """Set the trim portion for the current fit state.
This will update the trim index and burn index in the fit state."""
    async def shake_parameters() -> Any:
    async def shutdown() -> Any:
    async def start_fit(options: Any) -> Any:
    async def start_fit_thread(fitter_id: typing.Optional[str] = None, options: typing.Optional[typing.Dict[str, typing.Any]] = None, resume: bool = False) -> Any:
    async def stop_fit(wait: Any = True) -> Any:
        """Trigger the abort fit signal to the optimizer and wait for complete (or not)."""
    async def update_history_label(name: str, label: str) -> Any:
    async def update_serialized_problem(serialized: str, method: str = "dataclass", name: typing.Optional[str] = None) -> Any:
        """Update the current FitProblem without resetting the fit state.

Use this instead of ``set_serialized_problem`` when you want to resume
DREAM from the existing chain population rather than starting fresh.

Parameter-space changes (parameters added, removed, or renamed) are handled
automatically on the next ``start_fit_thread("dream", options, resume=True)``
call: ``DreamFit.solve`` detects the label mismatch and rebuilds the chain
state via ``_rebuild_mcmc_state`` before sampling begins.

Use ``set_serialized_problem()`` for a true cold start.

The *method* parameter accepts ``"dataclass"`` (default, safe),
``"pickle"``, ``"cloudpickle"``, or ``"dill"``.  The latter three enable
arbitrary code execution and carry the same caveat as
``set_serialized_problem``."""
    async def wait_for_fit(timeout: float = 30) -> Any:
        """Poll active_fit until empty (fit complete) or timeout."""
async def remote_bumps(url: str | None = None) -> bumps.webview.client.BumpsClient:
    """Open a client connection to a remote bumps server.

If *url* is not provided then start a new bumps server."""