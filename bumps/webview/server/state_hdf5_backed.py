import asyncio
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any, Literal, cast, IO
from collections import deque
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
    serialized = read_bytes_data(group, name)
    fitProblem = deserialize_problem(serialized, serializer) if serialized is not None else None
    return fitProblem

def write_json(group: 'Group', name: str, data):
    serialized = json.dumps(data) if data is not None else None
    dset = write_string(group, name, serialized.encode())
    return dset

def read_json(group: 'Group', name: str):
    serialized = read_string(group, name)
    try:
        # if JSON fails to load, then just return None
        result = json.loads(serialized.decode()) if serialized is not None else None
    except Exception:
        result = None
    return result

def write_ndarray(group: 'Group', name: str, data: Optional[np.ndarray], dtype=UNCERTAINTY_DTYPE):
    saved_data = data if data is not None else []
    return group.create_dataset(name, data=saved_data, dtype=dtype, compression=UNCERTAINTY_COMPRESSION)

def read_ndarray(group: 'Group', name: str):
    if not name in group:
        return None
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
    pathlist: Optional[List[str]] = None
    serializer: Optional[SERIALIZERS] = None
    filename: Optional[str] = None

    def write(self, parent: 'Group'):
        group = parent.require_group('problem')
        write_fitproblem(group, 'fitProblem', self.fitProblem, self.serializer)
        write_string(group, 'serializer', self.serializer)
        write_json(group, 'pathlist', self.pathlist)
        write_string(group, 'filename', self.filename)

    def read(self, parent: 'Group'):
        group = parent.require_group('problem')
        self.serializer = read_string(group, 'serializer')
        self.fitProblem = read_fitproblem(group, 'fitProblem', self.serializer)
        self.pathlist = read_json(group, 'pathlist')
        self.filename = read_string(group, 'filename')

class HistoryItem:
    problem: ProblemState
    fitting: Optional['FittingState']
    timestamp: str
    label: str
    chisq_str: str
    keep: bool


class History:
    store: List[HistoryItem]

    def __init__(self):
        self.store = []

    def write(self, parent: 'Group', include_uncertainty_state=True):
        group = parent.require_group('problem_history')
        for item in self.store:
            problem = item.problem
            fitting = item.fitting
            name = item.timestamp
            item_group = group.require_group(name)
            problem.write(item_group)
            fitting.write(item_group, include_uncertainty_state=include_uncertainty_state)
            item_group.attrs['chisq'] = item.chisq_str
            item_group.attrs['label'] = item.label
            item_group.attrs['keep'] = item.keep

    def read(self, parent: 'Group'):
        group = parent.get('problem_history', [])
        self.store = []
        for name in group:
            item = HistoryItem()
            item_group = group[name]
            item.problem = ProblemState()
            item.fitting = FittingState()
            item.problem.read(item_group)
            item.fitting.read(item_group)
            item.label = item_group.attrs['label']
            item.chisq_str = item_group.attrs['chisq']
            item.keep = item_group.attrs['keep']
            item.timestamp = name
            self.store.append(item)

    def remove_item(self, timestamp: str):
        self.store = [item for item in self.store if item.timestamp != timestamp]

    def add_item(self, item: HistoryItem):
        self.store.append(item)

    def list(self):
        return [dict(timestamp=item.timestamp, label=item.label, chisq_str=item.chisq_str) for item in self.store]
    
    def set_keep(self, timestamp: str, keep: bool):
        for item in self.store:
            if item.timestamp == timestamp:
                item.keep = keep
                return


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

    def write(self, parent: 'Group', include_uncertainty_state=True):
        group = parent.require_group('fitting')
        write_ndarray(group, 'population', self.population)
        uncertainty_state_storage = UncertaintyStateStorage()
        uncertainty_state = self.uncertainty_state
        if include_uncertainty_state and uncertainty_state is not None:
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


class FitterSettings:
    fitter: Optional[str] = None
    fitter_settings: Optional[Dict] = None

    def write(self, parent: 'Group'):
        group = parent.require_group('fitter_settings')
        write_string(group, 'fitter', self.fitter)
        write_json(group, 'fitter_settings', self.fitter_settings)

    def read(self, parent: 'Group'):
        group = parent.require_group('fitter_settings')
        self.fitter = read_string(group, 'fitter')
        self.fitter_settings = read_json(group, 'fitter_settings')


class State:
    # These attributes are ephemeral, not to be serialized/stored:
    hostname: str
    port: int
    parallel: int
    fit_thread: Optional['FitThread'] = None
    fit_abort: Optional[Event] = None
    fit_abort_event: Event
    fit_complete_event: Event
    fit_enabled: Event
    calling_loop: Optional[asyncio.AbstractEventLoop] = None
    session_file_name: Optional[str] = None

    # State to be stored:
    problem: ProblemState
    fitting: FittingState
    history: History
    topics: Dict['TopicNameType', 'deque[Dict]']

    def __init__(self, session_file_name: Optional[str] = None):
        self.problem = ProblemState()
        self.fitting = FittingState()
        self.history = History()
        self.fit_abort_event = Event()
        self.fit_complete_event = Event()
        self.topics = {
            "log": deque([]),
            "update_parameters": deque([], maxlen=1),
            "update_model": deque([], maxlen=1),
            "model_loaded": deque([], maxlen=1),
            "session_output_file": deque([], maxlen=1),
            "fit_active": deque([], maxlen=1),
            "convergence_update": deque([], maxlen=1),
            "uncertainty_update": deque([], maxlen=1),
            "fitter_settings": deque([], maxlen=1),
            "fitter_active": deque([], maxlen=1),
        }

    def __enter__(self):
        return self

    def setup_backing(self, session_file_name: Optional[str] = SESSION_FILE_NAME, read_only: bool = False ):
        if not read_only:
            self.session_file_name = session_file_name
        if session_file_name is not None:
            if Path(session_file_name).exists():
                self.read_session_file(session_file_name)
            else:
                self.save()

    def save_to_history(self, label: str, include_population: bool = False, include_uncertainty: bool = False, keep: bool = False):
        if self.problem.fitProblem is None:
            return
        item = HistoryItem()
        item.problem = deepcopy(self.problem)
        item.fitting = FittingState()
        if include_population:
            item.fitting.population = self.fitting.population
        if include_uncertainty:
            # uncertainty_state is not mutated, so we can safely use it here
            # rather than making a memory-hogging copy on first use.
            item.fitting.uncertainty_state = self.fitting.uncertainty_state

        item.timestamp = str(datetime.now())
        item.label = label
        item.chisq_str = item.problem.fitProblem.chisq_str()
        item.keep = keep
        self.history.add_item(item)

    def get_history(self):
        return dict(problem_history=self.history.list())

    def remove_history_item(self, timestamp: str):
        self.history.remove_item(timestamp)

    def reload_history_item(self, timestamp: str):
        for item in self.history.store:
            if item.timestamp == timestamp:
                print("problem found!", timestamp)
                problem_state = item.problem
                print('chisq of found item: ', problem_state.fitProblem.chisq_str())
                self.problem = deepcopy(problem_state)
                if item.fitting.population is not None:
                    self.fitting.population = item.fitting.population
                if item.fitting.uncertainty_state is not None:
                    self.fitting.uncertainty_state = item.fitting.uncertainty_state
                return
        raise ValueError(f"Could not find history item with timestamp {timestamp}")

    def save(self):
        if self.session_file_name is not None:
            self.write_session_file(self.session_file_name)

    def copy_session_file(self, session_copy_name: str):
        self.write_session_file(session_copy_name)

    def write_session_file(self, session_filename: str):
        tmp_fd, tmp_name = tempfile.mkstemp(dir=Path('.'))
        with os.fdopen(tmp_fd, 'w+b') as output_file:
            with h5py.File(output_file, 'w') as root_group:
                self.problem.write(root_group)
                self.fitting.write(root_group)
                self.write_topics(root_group)
        shutil.move(tmp_name, session_filename)

    def read_session_file(self, session_filename: str, read_problem: bool = True, read_fitstate: bool = True):
        try:
            with h5py.File(session_filename, 'r') as root_group:
                if read_problem:
                    self.problem.read(root_group)
                if read_fitstate:
                    self.fitting.read(root_group)
                self.read_topics(root_group)
        except Exception as e:
            logger.warning(f"could not load session file {session_filename} because of {e}")
            raise(e)

    def read_problem_from_session(self, session_filename: str):
        try:
            with h5py.File(session_filename, 'r') as root_group:
                self.problem.read(root_group)
        except Exception as e:
            logger.warning(f"could not load fitProblem from {session_filename} because of {e}")

    def read_fitstate_from_session(self, session_filename: str):
        try:
            with h5py.File(session_filename, 'r') as root_group:
                self.fitting.read(root_group)
        except Exception as e:
            logger.warning(f"could not load fit state from {session_filename} because of {e}")

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