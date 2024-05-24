import asyncio
from typing import TYPE_CHECKING, Optional, Dict, List, NewType, Tuple, TypedDict, Any, Literal, Union, cast, IO
from collections import deque
from dataclasses import dataclass, fields
import json
import shutil
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
from threading import Event
from bumps.serialize import serialize, deserialize, migrate
import h5py
import numpy as np

from bumps.dream.state import MCMCDraw
from .logger import logger

if TYPE_CHECKING:
    import bumps, bumps.fitproblem, bumps.dream.state
    from .webserver import TopicNameType
    from .fit_thread import FitThread
    from h5py import Group, Dataset


SESSION_FILE_NAME = "session.h5"
MAX_PROBLEM_SIZE = 100*1024*1024 # 10 MB problem max size
UNCERTAINTY_DTYPE = 'f'
MAX_LABEL_LENGTH = 1024
LABEL_DTYPE = f"|S{MAX_LABEL_LENGTH}"
COMPRESSION = 5
UNCERTAINTY_COMPRESSION = 5

SERIALIZERS = Literal['dataclass', 'pickle', 'dill']
SERIALIZER_EXTENSIONS = {
    'dataclass': 'json',
    'pickle': 'pickle',
    'dill': 'pickle'
}
DEFAULT_SERIALIZER: SERIALIZERS = "dill"

@dataclass(frozen=True)
class UNDEFINED_TYPE: pass

UNDEFINED = UNDEFINED_TYPE()

def serialize_problem(problem: 'bumps.fitproblem.FitProblem', method: SERIALIZERS):
    if method == 'dataclass':
        return json.dumps(serialize(problem)).encode()
    elif method == 'pickle':
        import pickle
        return pickle.dumps(problem)
    elif method == 'dill':
        import dill
        return dill.dumps(problem, recurse=True)

def deserialize_problem(serialized: bytes, method: SERIALIZERS):
    if method == 'dataclass':
        serialized_dict = json.loads(serialized)
        final_version, migrated = migrate(serialized_dict)
        return deserialize(migrated)
    elif method == 'pickle':
        import pickle
        return pickle.loads(serialized)
    elif method == 'dill':
        import dill
        return dill.loads(serialized)


def write_bytes_data(group: 'Group', name: str, data: bytes):
    saved_data = [data] if data is not None else []
    return group.create_dataset(name, data=np.void(saved_data), compression=COMPRESSION)

def read_bytes_data(group: 'Group', name: str):
    if not name in group:
        return UNDEFINED
    raw_data = group[name][()]
    size = raw_data.size
    if size is not None and size > 0:
        return raw_data.tobytes().rstrip(b'\x00')
    else:
        return None

def write_string(group: 'Group', name: str, data: str, encoding='utf-8'):
    saved_data = np.bytes_([data]) if data is not None else []
    return group.create_dataset(name, data=saved_data, compression=COMPRESSION, dtype=h5py.string_dtype(encoding=encoding))

def read_string(group: 'Group', name: str):
    if not name in group:
        return UNDEFINED
    raw_data = group[name][()]
    size = raw_data.size
    if size is not None and size > 0:
        return np.bytes_(raw_data.flat[0]).decode()
    else:
        return None

def write_fitproblem(group: 'Group', name: str, fitProblem: 'bumps.fitproblem.FitProblem', serializer: SERIALIZERS):
    serialized = serialize_problem(fitProblem, serializer) if fitProblem is not None else None
    dset = write_bytes_data(group, name, serialized)
    return dset

def read_fitproblem(group: 'Group', name: str, serializer: SERIALIZERS):
    if not name in group:
        return UNDEFINED
    serialized = read_bytes_data(group, name)
    fitProblem = deserialize_problem(serialized, serializer) if serialized is not None else None
    return fitProblem

def write_json(group: 'Group', name: str, data):
    serialized = json.dumps(data) if data is not None else None
    dset = write_string(group, name, serialized.encode())
    return dset

def read_json(group: 'Group', name: str):
    if not name in group:
        return UNDEFINED
    serialized = read_string(group, name)
    try:
        # if JSON fails to load, then just return None
        result = json.loads(serialized) if serialized is not None else None
    except Exception:
        result = None
    return result

def write_ndarray(group: 'Group', name: str, data: Optional[np.ndarray], dtype=UNCERTAINTY_DTYPE):
    saved_data = data if data is not None else []
    return group.create_dataset(name, data=saved_data, dtype=dtype, compression=UNCERTAINTY_COMPRESSION)

def read_ndarray(group: 'Group', name: str):
    if not name in group:
        return UNDEFINED
    raw_data = group[name][()]
    size = raw_data.size
    if size is not None and size > 0:
        return raw_data
    else:
        return None

class StringAttribute:
    @classmethod
    def serialize(value, obj=None):
        return json.dumps(value)

    @classmethod
    def deserialize(value, obj=None):
        return json.loads(value) if value else None

class ProblemState:
    fitProblem: Optional['bumps.fitproblem.FitProblem'] = None
    serializer: Optional[SERIALIZERS] = None

    def write(self, parent: 'Group'):
        group = parent.require_group('problem')
        write_fitproblem(group, 'fitProblem', self.fitProblem, self.serializer)
        write_string(group, 'serializer', self.serializer)
        # write_json(group, 'pathlist', self.pathlist)
        # write_string(group, 'filename', self.filename)

    def read(self, parent: 'Group'):
        group = parent.require_group('problem')
        self.serializer = read_string(group, 'serializer')
        self.fitProblem = read_fitproblem(group, 'fitProblem', self.serializer)
        # self.pathlist = read_json(group, 'pathlist')
        # self.filename = read_string(group, 'filename')
               

class UncertaintyStateStorage:
    AR: Optional['np.ndarray'] = None
    gen_draws: Optional['np.ndarray'] = None
    labels: Optional['np.ndarray'] = None
    thin_draws: Optional['np.ndarray'] = None
    gen_logp: Optional['np.ndarray'] = None
    thin_logp: Optional['np.ndarray'] = None
    thin_point: Optional['np.ndarray'] = None
    update_CR_weight: Optional['np.ndarray'] = None
    update_draws: Optional['np.ndarray'] = None
    good_chains: Optional['np.ndarray'] = None

    def write(self, parent: 'Group'):
        group = parent.require_group('uncertainty_state')
        for attrname in ['AR', 'gen_draws', 'thin_draws', 'gen_logp', 'thin_logp', 'thin_point', 'update_CR_weight', 'update_draws', 'good_chains']:
            write_ndarray(group, attrname, getattr(self, attrname), dtype=UNCERTAINTY_DTYPE)
        write_ndarray(group, 'labels', self.labels, dtype=LABEL_DTYPE)

    def read(self, parent: 'Group'):
        group = parent['uncertainty_state']
        for attrname in ['AR', 'gen_draws', 'labels', 'thin_draws', 'gen_logp', 'thin_logp', 'thin_point', 'update_CR_weight', 'update_draws', 'good_chains']:
            setattr(self, attrname, read_ndarray(group, attrname))

class FittingState:
    population: Optional[List] = None
    uncertainty_state: Optional['bumps.dream.state.MCMCDraw'] = None

    def write(self, parent: 'Group'):
        group = parent.require_group('fitting')
        write_ndarray(group, 'population', self.population)
        uncertainty_state_storage = UncertaintyStateStorage()
        uncertainty_state = self.uncertainty_state
        if uncertainty_state is not None:
            write_uncertainty_state(uncertainty_state, uncertainty_state_storage)
            uncertainty_state_storage.write(group)

    def read(self, parent: 'Group'):
        group = parent['fitting']
        population = read_ndarray(group, 'population')
        self.population = population
        if 'uncertainty_state' in group:
            uncertainty_state_storage = UncertaintyStateStorage()
            uncertainty_state_storage.read(group)
            self.uncertainty_state = read_uncertainty_state(uncertainty_state_storage)


class State:
    # These attributes are ephemeral, not to be serialized/stored:
    hostname: str
    port: int
    parallel: int
    fit_thread: Optional['FitThread'] = None
    fit_abort: Optional[Event] = None
    fit_abort_event: Event
    fit_complete_event: Event
    fit_uncertainty_final: Event
    fit_enabled: Event
    calling_loop: Optional[asyncio.AbstractEventLoop] = None

    # State to be stored:
    problem: ProblemState
    fitting: FittingState
    topics: Dict['TopicNameType', 'deque[Dict]']
    shared: 'SharedState'

    def __init__(self):
        self.problem = ProblemState()
        self.fitting = FittingState()
        self.fit_abort_event = Event()
        self.fit_complete_event = Event()
        self.fit_uncertainty_final = Event()
        self.topics = {
            "log": deque([]),

        }
        self.shared = SharedState()

    def __enter__(self):
        return self

    def setup_backing(self, session_file_name: str, session_pathlist: List[str], read_only: bool = False ):
        if not read_only:
            self.shared.session_output_file = dict(filename=session_file_name, pathlist=session_pathlist)
        if session_file_name is not None:
            if Path(session_file_name).exists():
                self.read_session_file(session_file_name)
            else:
                self.save()
    
    def autosave(self):
        if self.shared.autosave_session:
            self.save()

    def save(self):
        if self.shared.session_output_file not in [None, UNDEFINED]:
            pathlist = self.shared.session_output_file['pathlist']
            filename = self.shared.session_output_file['filename']
            full_path = Path(*pathlist) / filename
            self.write_session_file(full_path)

    def write_session_file(self, session_fullpath: str):
        # Session filename is assumed to be a full path
        tmp_fd, tmp_name = tempfile.mkstemp(dir=Path(session_fullpath).parent, prefix=Path(session_fullpath).name, suffix='.tmp')
        with os.fdopen(tmp_fd, 'w+b') as output_file:
            with h5py.File(output_file, 'w') as root_group:
                self.problem.write(root_group)
                self.fitting.write(root_group)
                self.write_topics(root_group)
                self.shared.write(root_group)
        shutil.move(tmp_name, session_fullpath)
        os.chmod(session_fullpath, 0o644)

    def read_session_file(self, session_fullpath: str, read_problem: bool = True, read_fitstate: bool = True):
        try:
            with h5py.File(session_fullpath, 'r') as root_group:
                if read_problem:
                    self.problem.read(root_group)
                if read_fitstate:
                    self.fitting.read(root_group)
                self.read_topics(root_group)
                self.shared.read(root_group)
        except Exception as e:
            logger.warning(f"could not load session file {session_fullpath} because of {e}")

    def read_problem_from_session(self, session_fullpath: str):
        try:
            with h5py.File(session_fullpath, 'r') as root_group:
                self.problem.read(root_group)
        except Exception as e:
            logger.warning(f"could not load fitProblem from {session_fullpath} because of {e}")

    def read_fitstate_from_session(self, session_fullpath: str):
        try:
            with h5py.File(session_fullpath, 'r') as root_group:
                self.fitting.read(root_group)
        except Exception as e:
            logger.warning(f"could not load fit state from {session_fullpath} because of {e}")

    def write_topics(self, parent: 'Group'):
        group = parent.require_group('topics')
        for topic, messages in self.topics.items():
            write_json(group, topic, list(messages))

    def read_topics(self, parent: 'Group'):
        group = parent.require_group('topics')
        for topic in group:
            topic_data = read_json(group, topic)
            topic_data = np.array([topic_data]).flatten()
            if topic_data is not None:
                self.topics[topic].extend(topic_data)

    def get_last_message(self, topic: 'TopicNameType'):
        return self.topics[topic][-1] if len(self.topics[topic]) > 0 else None

    def cleanup(self):
        pass

    def __del__(self):
        self.cleanup()

    async def async_cleanup(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cleanup()


def write_uncertainty_state(state: 'MCMCDraw', storage: UncertaintyStateStorage):
        # Build 2-D data structures
        storage.gen_draws, storage.gen_logp = state.logp(full=True)
        _, storage.AR = state.acceptance_rate()

        storage.thin_draws, storage.thin_point, storage.thin_logp = state.chains()
        storage.update_draws, storage.update_CR_weight = state.CR_weight()
        storage.labels = np.array(state.labels, dtype=LABEL_DTYPE)
        good_chains = state._good_chains
        storage.good_chains = None if isinstance(good_chains, slice) else good_chains

def read_uncertainty_state(loaded: UncertaintyStateStorage, skip=0, report=0, derived_vars=0):

    # Guess dimensions
    Ngen = loaded.gen_draws.shape[0]
    thinning = 1
    Nthin, Npop, Nvar = loaded.thin_point.shape
    Nupdate, Ncr = loaded.update_CR_weight.shape
    Nthin -= skip
    good_chains = loaded.good_chains

    # Create empty draw and fill it with loaded data
    state = MCMCDraw(0, 0, 0, 0, 0, 0, thinning)
    state.draws = Ngen * Npop
    state.labels = [label.decode() for label in loaded.labels]
    state.generation = Ngen
    state._gen_index = 0
    state._gen_draws = loaded.gen_draws
    state._gen_acceptance_rate = loaded.AR
    state._gen_logp = loaded.gen_logp
    state.thinning = thinning
    state._thin_count = Ngen//thinning
    state._thin_index = 0
    state._thin_draws = loaded.thin_draws
    state._thin_logp = loaded.thin_logp
    state._thin_point = loaded.thin_point
    state._gen_current = state._thin_point[-1].copy()
    state._update_count = Nupdate
    state._update_index = 0
    state._update_draws = loaded.update_draws
    state._update_CR_weight = loaded.update_CR_weight
    state._outliers = []

    bestidx = np.unravel_index(np.argmax(loaded.thin_logp), loaded.thin_logp.shape)
    state._best_logp = loaded.thin_logp[bestidx]
    state._best_x = loaded.thin_point[bestidx]
    state._best_gen = 0

    state._good_chains = slice(None, None) if good_chains is None else good_chains.astype(int)

    return state

class ActiveFit(TypedDict):
    fitter_id: str
    options: Dict[str, Any]
    num_steps: int

class FileInfo(TypedDict):
    filename: str
    pathlist: List[str]
    
Timestamp = NewType('Timestamp', str)

@dataclass
class SharedState:
    updated_convergence: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    updated_uncertainty: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    updated_parameters: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    updated_model: Union[UNDEFINED_TYPE, Timestamp] = UNDEFINED
    selected_fitter: Union[UNDEFINED_TYPE, str] = UNDEFINED
    fitter_settings: Union[UNDEFINED_TYPE, Dict[str, Dict]] = UNDEFINED
    active_fit: Union[UNDEFINED_TYPE, ActiveFit] = UNDEFINED
    model_file: Union[UNDEFINED_TYPE, FileInfo] = UNDEFINED
    model_loaded: Union[UNDEFINED_TYPE, bool] = UNDEFINED
    session_output_file: Union[UNDEFINED_TYPE, FileInfo] = UNDEFINED
    autosave_session: Union[UNDEFINED_TYPE, bool] = UNDEFINED
    
    async def notify(self, name, value):
        pass

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.notify(name, value))

    async def set(self, name, value):
        super().__setattr__(name, value)
        await self.notify(name, value)

    async def get(self, name):
        return getattr(self, name, UNDEFINED)

    def write(self, parent: 'Group'):
        group = parent.require_group('shared')
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not UNDEFINED:
                write_json(group, field.name, value)

    def read(self, parent: 'Group'):
        group = parent.get('shared')
        if group is None:
            return
        for field in fields(self):
            setattr(self, field.name, read_json(group, field.name))