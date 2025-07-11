import asyncio
import threading
from copy import deepcopy
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Optional,
    Dict,
    Tuple,
    List,
    NewType,
    TypedDict,
    Any,
    Literal,
    Union,
)
from collections import deque
from dataclasses import dataclass, field, fields
import json
import shutil
import os
import tempfile
from pathlib import Path
import pickle

import h5py
import numpy as np
from numpy.typing import NDArray
import dill

from bumps import __version__
from bumps.serialize import serialize, deserialize
from bumps.serialize import serialize_bytes, deserialize_bytes
from bumps.util import get_libraries
from .logger import logger

from .fit_options import lookup_fitter, DEFAULT_FITTER_ID

if TYPE_CHECKING:
    import bumps
    import bumps.fitproblem
    import bumps.dream.state
    from .webserver import TopicNameType
    from .fit_thread import FitThread
    from h5py import Group
from bumps.mapper import BaseMapper


SESSION_FILE_NAME = "session.h5"
ARRAY_COMPRESSION = 5
COMPRESSION = 9
# MAX_PROBLEM_SIZE = 100 * 1024 * 1024  # 100 MBi problem max size [unused]

SERIALIZERS = Literal["dataclass", "pickle", "dill"]
SERIALIZER_EXTENSIONS = {"dataclass": "json", "pickle": "pickle", "dill": "pickle"}
DEFAULT_SERIALIZER: SERIALIZERS = "dill"


@dataclass(frozen=True)
class UNDEFINED_TYPE:
    pass


UNDEFINED = UNDEFINED_TYPE()


def now_string():
    return f"{datetime.now().timestamp():.6f}"


def serialize_problem(problem: "bumps.fitproblem.FitProblem", method: SERIALIZERS) -> Union[str, bytes]:
    if method == "dataclass":
        return json.dumps(serialize(problem))
    elif method == "pickle":
        return serialize_bytes(pickle.dumps(problem))
    elif method == "dill":
        return serialize_bytes(dill.dumps(problem, recurse=True))
    else:
        raise ValueError(f"Unknown serialization method: {method}")


def deserialize_problem(serialized: str, method: SERIALIZERS) -> "bumps.fitproblem.FitProblem":
    if method == "dataclass":
        serialized_dict = json.loads(serialized)
        return deserialize(serialized_dict, migration=True)
    elif method == "pickle":
        return pickle.loads(deserialize_bytes(serialized))
    elif method == "dill":
        return dill.loads(deserialize_bytes(serialized))
    else:
        raise ValueError(f"Unknown serialization method: {method}")


def serialize_problem_bytes(problem: "bumps.fitproblem.FitProblem", method: SERIALIZERS) -> bytes:
    if method == "dataclass":
        return json.dumps(serialize(problem)).encode()
    elif method == "pickle":
        return pickle.dumps(problem)
    elif method == "dill":
        return dill.dumps(problem, recurse=True)
    else:
        raise ValueError(f"Unknown serialization method: {method}")


def deserialize_problem_bytes(serialized: bytes, method: SERIALIZERS) -> "bumps.fitproblem.FitProblem":
    if method == "dataclass":
        serialized_dict = json.loads(serialized.decode())
        return deserialize(serialized_dict, migration=True)
    elif method == "pickle":
        return pickle.loads(serialized)
    elif method == "dill":
        return dill.loads(serialized)
    else:
        raise ValueError(f"Unknown serialization method: {method}")


def write_bytes(group: "Group", name: str, data: bytes):
    saved_data = [data] if data is not None else []
    return group.create_dataset(name, data=np.void(saved_data), compression=COMPRESSION)


def read_bytes(group: "Group", name: str):
    if name not in group:
        return UNDEFINED
    raw_data = group[name][()]
    size = raw_data.size
    if size is not None and size > 0:
        return raw_data[0].tobytes().rstrip(b"\x00")
    else:
        return None


def write_string(group: "Group", name: str, data: str, encoding="utf-8"):
    if data is None:
        return group.create_dataset(name, data="")
    # saved_data = np.bytes_([data]) if data is not None else []
    dtype = h5py.string_dtype(encoding=encoding, length=len(data))
    saved_data = np.array([data], dtype=dtype) if data is not None else []
    # print(f"write_string {dtype=}")
    return group.create_dataset(name, data=saved_data, compression=COMPRESSION, dtype=dtype)


def read_string(group: "Group", name: str):
    if name not in group:
        return UNDEFINED
    raw_data = group[name][()]
    size = raw_data.size
    if size is not None and size > 0:
        return np.bytes_(raw_data.flat[0]).decode()
    else:
        return None


def write_fitproblem(group: "Group", name: str, fitProblem: "bumps.fitproblem.FitProblem", method: SERIALIZERS):
    encoding = str if method == "dataclass" else bytes
    if encoding is bytes:
        serialized = serialize_problem_bytes(fitProblem, method) if fitProblem is not None else None
        dset = write_bytes(group, name, serialized)
    else:
        serialized = serialize_problem(fitProblem, method) if fitProblem is not None else None
        dset = write_string(group, name, serialized)
    return dset


def read_fitproblem(group: "Group", name: str, method: SERIALIZERS) -> "bumps.fitproblem.FitProblem":
    if name not in group:
        return UNDEFINED
    if group[name].dtype.kind == "V":
        # Old encoding stored bytes directly
        serialized = read_bytes(group, name)
        fitProblem = deserialize_problem_bytes(serialized, method) if serialized is not None else None
    else:
        # New encode uses base64 to encode bytes to string
        serialized = read_string(group, name)
        fitProblem = deserialize_problem(serialized, method) if serialized is not None else None
    return fitProblem


def write_json(group: "Group", name: str, data):
    dset = write_string(group, name, json.dumps(data))
    return dset


def read_json(group: "Group", name: str):
    if name not in group:
        return UNDEFINED
    serialized = read_string(group, name)
    try:
        # if JSON fails to load, then just return None
        result = json.loads(serialized) if serialized is not None else None
    except Exception:
        result = None
    return result


def write_ndarray(group: "Group", name: str, data: Optional[NDArray]):
    if data is None:
        data = []
        dtype = "d"
        compression = 0
    else:
        dtype = data.dtype
        compression = ARRAY_COMPRESSION
    return group.create_dataset(name, data=data, dtype=dtype, compression=compression)


def read_ndarray(group: "Group", name: str):
    if name not in group:
        return UNDEFINED
    raw_data = group[name][()]
    size = raw_data.size
    if size is not None and size > 0:
        return raw_data
    else:
        return None


def read_version(group: "Group"):
    version_string = group.attrs.get("version", "0.0")
    version = tuple(int(v) for v in version_string.split("."))
    return version


def write_version(group: "Group", version: Tuple[int]):
    version_string = ".".join(str(v) for v in version)
    group.attrs["version"] = version_string


class StringAttribute:
    @classmethod
    def serialize(value, obj=None):
        return json.dumps(value)

    @classmethod
    def deserialize(value, obj=None):
        return json.loads(value) if value else None


@dataclass
class ProblemState:
    fitProblem: Optional["bumps.fitproblem.FitProblem"] = None
    serializer: Optional[SERIALIZERS] = None

    def write(self, parent: "Group"):
        group = parent.require_group("problem")
        write_fitproblem(group, "fitProblem", self.fitProblem, method=self.serializer)
        write_string(group, "serializer", self.serializer)
        write_json(group, "libraries", get_libraries(self.fitProblem))
        # write_json(group, 'pathlist', self.pathlist)
        # write_string(group, 'filename', self.filename)

    def read(self, parent: "Group"):
        group = parent["problem"]
        self.serializer = read_string(group, "serializer")
        self.fitProblem = read_fitproblem(group, "fitProblem", method=self.serializer)
        # self.pathlist = read_json(group, 'pathlist')
        # self.filename = read_string(group, 'filename')


class HistoryItem:
    problem: ProblemState
    fitting: Optional["FitResult"]
    timestamp: str
    label: str
    chisq_str: str
    keep: bool


class History:
    store: Dict[str, HistoryItem]

    def __init__(self):
        self.store = {}

    def get_item(self, name: Union[str, UNDEFINED_TYPE, None], default=None):
        return self.store.get(name, default)

    # def get_item(self, timestamp: Union[str, UNDEFINED_TYPE, None], default=None):
    #    for item in self.store:
    #         if item.timestamp == timestamp:
    #             return item
    #     return default

    def write(self, parent: "Group", include_fit_state=True):
        group = parent.require_group("problem_history")
        for name, item in self.store.items():
            problem = item.problem
            fitting = item.fitting
            item_group = group.require_group(name)
            problem.write(item_group)
            fitting.write(item_group, include_fit_state=include_fit_state)
            item_group.attrs["chisq"] = item.chisq_str
            item_group.attrs["label"] = item.label
            item_group.attrs["keep"] = item.keep
            item_group.attrs["timestamp"] = item.timestamp
        return group

    def read(self, parent: "Group"):
        group = parent.get("problem_history", {})
        self.store = {}
        for name in group:
            item = HistoryItem()
            item_group = group[name]
            item.problem = ProblemState()
            item.fitting = FitResult()
            item.problem.read(item_group)
            item.fitting.read(item_group)
            item.label = item_group.attrs["label"]
            item.chisq_str = item_group.attrs["chisq"]
            # keep is a boolean, but h5py returns np.bool_ which is not JSON serializable
            item.keep = bool(item_group.attrs["keep"])
            # if there is no timestamp attribute, then it was created before we had a separate name
            item.timestamp = item_group.attrs.get("timestamp", name)
            self.store[name] = item

    def remove_item(self, name: str):
        self.store.pop(name)

    def prune(self, target_length: int):
        # remove oldest items with keep=False until the length is target_length
        num_to_remove = len(self.store) - target_length
        if num_to_remove <= 0:
            return
        names_to_remove = []
        for name, item in self.store.items():
            if not item.keep:
                names_to_remove.append(name)
                num_to_remove -= 1
                if num_to_remove == 0:
                    break
        for name in reversed(names_to_remove):
            self.store.pop(name)

    def _get_unique_name(self, timestamp: str):
        name = timestamp
        counter = 1
        while name in self.store:
            name = f"{timestamp}-{counter}"
            counter += 1
        return name

    def add_item(self, item: HistoryItem, target_length: int):
        self.prune(target_length)
        stored_name = self._get_unique_name(item.timestamp)
        self.store[stored_name] = item
        return stored_name

    def list(self):
        return [
            dict(
                timestamp=item.timestamp,
                label=item.label,
                chisq_str=item.chisq_str,
                keep=item.keep,
                has_convergence=(item.fitting.convergence is not None),
                has_uncertainty=hasattr(item.fitting.fit_state, "draw"),
                name=name,
            )
            for name, item in self.store.items()
        ]

    def set_keep(self, name: str, keep: bool):
        self.store[name].keep = keep

    def update_label(self, name: str, label: str):
        self.store[name].label = label


# TODO: Where do derived expressions and nuisance parameters live? problem or results?
# TODO: Use uncertainties package with cov for derived parameters from amoeba
# TODO: Showing error table requires parameter labels; get them from fit problem?


@dataclass
class FitResult:
    # TODO: chisq, dof, and {model: (chisq, dof)} separate from fx=nllf+constraints
    # TODO: Model specific dof is difficult because of shared parameters. Just use #points?
    # TODO: Rename fitter_id to method throughout?
    # TODO: Should the best point include all available parameters in the model (fitted and fixed)?
    # TODO: Save labels in fit results so we don't need to walk the problem definition?
    # TODO: Save the initial value in the problem so users can reset after a fit?
    # TODO: Add the following to FitResult:
    method: str = DEFAULT_FITTER_ID  # => shared.selected_fitter
    """Fitting method"""
    options: Dict[str, Any] = field(default_factory=dict)  # => shared.fitter_settings
    """Options used to run the fitters"""
    # x0: NDArray
    # """Initial value"""
    # x: NDArray  # Currently resides in problem definition
    # """Best point"""
    # TODO: include dx, cov and entropy?
    # TODO: these are odd men out: they are only available on completion
    # dx: Optional[NDArray]
    # """Uncertainty from derivative if fit is complete"""
    # fx: float  # nllf maybe including constraints and penalties
    # """Best value"""
    # TODO: Maybe add maxsteps (it could be guessed from options)
    # step: int  # Should equal the length of the population, so unneeded
    # """Number of optimizer steps taken"""
    # run_time: float # seconds
    # """Number of seconds that the fit was run before completion/abort/timeout"""
    # cpu_hours: float
    # """Total number of cpu hours for the fit (=num_processors*wall_time/3600)"""
    # TODO: display completion status in history tab
    # status: str
    # """Fit status: active, converged, timeout, maxiter, abort, failed"
    convergence: Optional[List] = None
    """List of best or (best, min, -1sigma, median, +1sigma, max) for the population at each step of the fit."""
    fit_state: Any = None
    """Fit state for resume, and for sampling from Monte Carlo fitters."""

    def write(self, parent: "Group", include_fit_state=True):
        fitting_group = parent.require_group("fitting")
        write_version(fitting_group, (1, 0))
        write_string(fitting_group, "method", self.method)
        write_json(fitting_group, "options", self.options)
        write_ndarray(fitting_group, "convergence", self.convergence)
        if self.fit_state is not None and include_fit_state:
            fitter = lookup_fitter(self.method)
            if hasattr(fitter, "h5dump"):
                state_group = fitting_group.require_group("fit_state")
                fitter.h5dump(state_group, self.fit_state)

    def read(self, parent: "Group"):
        fitting_group = parent["fitting"]
        version = read_version(fitting_group)
        # Note: fitter h5load needs to deal with its own versioning
        if version == (1, 0):
            self.method = read_string(fitting_group, "method")
            self.options = read_json(fitting_group, "options")
            self.convergence = read_ndarray(fitting_group, "convergence")
            if "fit_state" in fitting_group:
                state_group = fitting_group["fit_state"]
                fitter = lookup_fitter(self.method)  # shouldn't raise ValueError
                self.fit_state = fitter.h5load(state_group)
            else:
                self.fit_state = None
        else:
            # Pre 1.0 fit result
            self.convergence = read_ndarray(fitting_group, "population")
            self.options = {}  # options
            if "uncertainty_state" in fitting_group:
                self.method = "dream"
                state_group = fitting_group["uncertainty_state"]
                fitter = lookup_fitter(self.method)
                self.fit_state = fitter.h5load(state_group)
            else:
                self.method = DEFAULT_FITTER_ID
                self.fit_state = None


class State:
    # These attributes are ephemeral, not to be serialized/stored:
    app_name: str = "bumps"
    app_version: str = __version__
    client_path: Path = Path(__file__).parent.parent / "client"
    hostname: str
    port: int
    parallel: int = 0
    fit_thread: Optional["FitThread"] = None
    fit_abort_event: threading.Event
    """Cleared before the fit and set on Stop button or Ctrl-C to end the fit."""
    fit_complete_event: asyncio.Event
    """Cleared before the fit starts and set when the fit is complete and saved."""
    # fit_complete_future: asyncio.Future
    shutdown_on_fit_complete: bool = False
    """Used to implement the --exit option to halt server on completion."""
    # fit_enabled: Event
    calling_loop: Optional[asyncio.AbstractEventLoop] = None
    base_path: str = ""
    console_update_interval: int = 0  # seconds (float would work too, but unnecessary)

    # State to be stored:
    problem: ProblemState
    fitting: FitResult
    history: History
    topics: Dict["TopicNameType", "deque[Dict]"]
    shared: "SharedState"
    mapper: Optional[BaseMapper] = None

    def __init__(self):
        self.problem = ProblemState()
        self.fitting = FitResult()
        self.history = History()
        self.fit_abort_event = threading.Event()  # initially unset
        self.fit_complete_event = asyncio.Event()
        self.fit_complete_event.set()  # The program starts out not waiting for a fit
        self.topics = {
            "log": deque([]),
        }
        self.shared = SharedState()

    def __enter__(self):
        return self

    def setup_backing(self, session_file_name: str, session_pathlist: List[str], read_only: bool = False):
        if not read_only:
            self.shared.session_output_file = dict(filename=session_file_name, pathlist=session_pathlist)
        if session_file_name is not None:
            full_path = Path(*session_pathlist) / session_file_name
            if full_path.exists():
                self.read_session_file(full_path)
            else:
                self.save()

    def save_to_history(self, label: str, keep: bool = False) -> str:
        if self.problem.fitProblem is None:
            return
        item = HistoryItem()
        item.problem = deepcopy(self.problem)
        item.fitting = deepcopy(self.fitting)
        item.timestamp = str(datetime.now())
        item.label = label
        item.chisq_str = item.problem.fitProblem.chisq_str()
        item.keep = keep
        stored_name = self.history.add_item(item, self.shared.autosave_history_length - 1)
        self.shared.updated_history = now_string()
        return stored_name

    def get_history(self):
        return dict(problem_history=self.history.list())

    def remove_history_item(self, name: str):
        if self.shared.active_history == name:
            self.shared.active_history = None
        self.history.remove_item(name)

    def reload_history_item(self, name: str):
        item = self.history.get_item(name, None)
        if item is not None:
            self.problem = deepcopy(item.problem)
            self.fitting = item.fitting
            self.shared.active_history = name
            self.shared.updated_model = now_string()
            self.shared.updated_parameters = now_string()
            self.shared.custom_plots_available = get_custom_plots_available(self.problem.fitProblem)
            # These are called only to trigger the update signals...
            # the convergence and fit_state will be unchanged by the calls below.
            self.set_convergence(item.fitting.convergence)
            self.set_fit_state(item.fitting.fit_state, item.fitting.method)

    def reset_fitstate(self, copy: bool = False):
        """
        Unlink the fitting state from a history item:
        (this action to be taken when fitProblem object is modified so that
         it is no longer compatible with fit results)
        """
        if copy:
            self.fitting = deepcopy(self.fitting)
            # print(f"reset_fitstate {copy}: keeping {self.fitting.method} with {self.fitting.fit_state} and convergence={self.fitting.convergence is not None}")
        else:
            # print(f"reset_fitstate {copy}: keeping {self.fitting.method}")
            self.fitting = FitResult(
                method=self.shared.selected_fitter,
                options=self.shared.fitter_settings[self.shared.selected_fitter]["settings"],
            )
            # These are called only to trigger the update signals...
            self.set_convergence(None)
            self.set_fit_state(None)
        self.shared.active_history = None

    def set_convergence(self, convergence):
        # print("setting convergence", convergence is not None)
        self.fitting.convergence = convergence
        self.shared.updated_convergence = now_string()
        self.shared.convergence_available = convergence is not None

    def set_fit_state(self, fit_state, method=None):
        self.fitting.fit_state = fit_state
        self.shared.updated_uncertainty = now_string()
        self.shared.uncertainty_available = dict(
            available=hasattr(fit_state, "draw"),
            num_points=getattr(fit_state, "Nsamples", 0),
        )
        self.shared.resumable = method if fit_state is not None else None

    def autosave(self):
        if self.shared.autosave_session:
            self.save()

    def save(self):
        if self.shared.session_output_file not in [None, UNDEFINED]:
            pathlist = self.shared.session_output_file["pathlist"]
            filename = self.shared.session_output_file["filename"]
            full_path = Path(*pathlist) / filename
            self.write_session_file(full_path)

    def write_session_file(self, session_fullpath: str):
        # Session filename is assumed to be a full path
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=Path(session_fullpath).parent, prefix=Path(session_fullpath).name, suffix=".tmp"
        )
        with os.fdopen(tmp_fd, "w+b") as output_file:
            with h5py.File(output_file, "w") as root_group:
                self.problem.write(root_group)
                history_group = self.history.write(root_group)
                if self.shared.active_history is not None:
                    active_history_group = history_group.get(self.shared.active_history)
                    # make a hard link instead of writing the fitting state
                    root_group["fitting"] = active_history_group["fitting"]
                else:
                    # no active history item, so write the fitting state
                    self.fitting.write(root_group)
                self.write_topics(root_group)
                self.shared.write(root_group)
        shutil.move(tmp_name, session_fullpath)
        os.chmod(session_fullpath, 0o644)

    def read_session_file(self, session_fullpath: str, read_problem: bool = True, read_fitstate: bool = True):
        try:
            with h5py.File(session_fullpath, "r") as root_group:
                if read_problem:
                    self.problem.read(root_group)
                self.history.read(root_group)
                self.shared.read(root_group)
                if read_fitstate:
                    active_item = self.history.get_item(self.shared.active_history, None)
                    if active_item is not None:
                        self.fitting = active_item.fitting
                    else:
                        self.fitting.read(root_group)
                self.read_topics(root_group)
        except Exception as e:
            # logger.exception(e)
            logger.warning(f"could not load session file {session_fullpath} because of {e}")

    def read_problem_from_session(self, session_fullpath: str):
        try:
            with h5py.File(session_fullpath, "r") as root_group:
                self.problem.read(root_group)
        except Exception as e:
            # logger.exception(e)
            logger.warning(f"could not load fitProblem from {session_fullpath} because of {e}")

    def read_fitstate_from_session(self, session_fullpath: str):
        try:
            with h5py.File(session_fullpath, "r") as root_group:
                self.fitting.read(root_group)
        except Exception as e:
            # logger.exception(e)
            logger.warning(f"could not load fit state from {session_fullpath} because of {e}")

    def write_topics(self, parent: "Group"):
        group = parent.require_group("topics")
        for topic, messages in self.topics.items():
            write_json(group, topic, list(messages))

    def read_topics(self, parent: "Group"):
        group = parent.require_group("topics")
        for topic in group:
            topic_data = read_json(group, topic)
            topic_data = np.array([topic_data]).flatten()
            if topic_data is not None and topic in self.topics:
                self.topics[topic].extend(topic_data)

    def get_last_message(self, topic: "TopicNameType"):
        return self.topics[topic][-1] if len(self.topics[topic]) > 0 else None

    def cleanup(self):
        pass

    def __del__(self):
        self.cleanup()

    async def async_cleanup(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cleanup()


class ActiveFit(TypedDict):
    fitter_id: str
    options: Dict[str, Any]
    num_steps: int
    step: int
    chisq: str
    value: float


class FileInfo(TypedDict):
    filename: str
    pathlist: List[str]


class UncertaintyAvailable(TypedDict):
    available: bool
    num_points: int


class CustomPlotsAvailable(TypedDict):
    parameter_based: bool
    uncertainty_based: bool


Timestamp = NewType("Timestamp", str)


def get_custom_plots_available(problem: "bumps.fitproblem.FitProblem"):
    output = {"parameter_based": False, "uncertainty_based": False}
    for model in problem.models:
        if hasattr(model, "webview_plots"):
            for plot_title, plot_info in model.webview_plots.items():
                if plot_info.get("change_with", None) == "uncertainty":
                    output["uncertainty_based"] = True
                else:
                    output["parameter_based"] = True
    return output


@dataclass
class SharedState:
    updated_convergence: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    updated_uncertainty: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    updated_parameters: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    updated_model: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    updated_history: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    selected_fitter: Union[UNDEFINED_TYPE, str] = "amoeba"
    fitter_settings: Union[UNDEFINED_TYPE, Dict[str, Dict]] = UNDEFINED
    active_fit: Union[UNDEFINED_TYPE, ActiveFit] = UNDEFINED
    model_file: Union[UNDEFINED_TYPE, FileInfo] = UNDEFINED
    model_loaded: Union[UNDEFINED_TYPE, bool] = UNDEFINED
    session_output_file: Union[UNDEFINED_TYPE, FileInfo] = UNDEFINED
    autosave_session: bool = False
    autosave_session_interval: int = 300  # seconds
    autosave_history: bool = True
    autosave_history_length: int = 10
    uncertainty_available: Union[UNDEFINED_TYPE, UncertaintyAvailable] = UNDEFINED
    convergence_available: Union[UNDEFINED_TYPE, bool] = UNDEFINED
    resumable: Union[UNDEFINED_TYPE, str, None] = None
    custom_plots_available: Union[UNDEFINED_TYPE, CustomPlotsAvailable] = UNDEFINED
    active_history: Union[UNDEFINED_TYPE, str, None] = UNDEFINED  # name of the active history item

    _not_reloaded = ["active_fit", "autosave_session", "session_output_file", "_notification_callbacks"]
    _notification_callbacks: Dict[str, Callable[[str, Any], Awaitable[None]]] = field(default_factory=dict)

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # no event loop running, so no need to notify
            return
        if hasattr(self, "_notification_callbacks"):
            for callback in self._notification_callbacks.values():
                loop.create_task(callback(name, value))

    async def set(self, name, value):
        super().__setattr__(name, value)
        for callback in self._notification_callbacks.values():
            await callback(name, value)

    async def notify(self, name, value=None):
        value = await self.get(name)
        for callback in self._notification_callbacks.values():
            await callback(name, value)

    async def get(self, name):
        return getattr(self, name, UNDEFINED)

    def write(self, parent: "Group"):
        group = parent.require_group("shared")
        for f in fields(self):
            if f.name not in self._not_reloaded:
                value = getattr(self, f.name)
                if value is not UNDEFINED:
                    write_json(group, f.name, value)

    def read(self, parent: "Group"):
        group = parent.get("shared")
        if group is None:
            return
        for f in fields(self):
            if f.name not in self._not_reloaded:
                value = read_json(group, f.name)
                if value is not UNDEFINED:
                    setattr(self, f.name, value)
