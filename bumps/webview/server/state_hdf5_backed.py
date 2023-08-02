import asyncio
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any, Literal, cast, IO
import json
import pickle
from queue import Queue
from threading import Event
from bumps.serialize import from_dict, from_dict_threaded, to_dict
import numpy as np
import warnings
import io

import bumps.fitproblem
from bumps.dream.state import MCMCDraw

if TYPE_CHECKING:
    from .api import TopicNameType
    from h5py import Group, Dataset, File
    from .fit_thread import FitThread

# slow, small:
COMPRESSION = 9
MAX_TOPIC_MESSAGE = 1024*100 # 100k
MAX_PROBLEM_SIZE = 10*1024*1024 # 10 MB problem max size
SESSION_FILE_NAME = "session.h5"
HDF5_VERSION = "v112"

CACHE_MISS = object()
SERIALIZERS = Literal['dataclass', 'pickle', 'dill']
SERIALIZER_EXTENSIONS = {
    'dataclass': 'json',
    'pickle': 'pickle',
    'dill': 'pickle'
}
DEFAULT_SERIALIZER: SERIALIZERS = "dill"

def serialize_problem(problem: bumps.fitproblem.FitProblem, method: SERIALIZERS):
    if method == 'dataclass':
        return json.dumps(to_dict(problem)).encode()
    elif method == 'pickle':
        return pickle.dumps(problem)
    elif method == 'dill':
        import dill
        return dill.dumps(problem)

def deserialize_problem(serialized: bytes, method: SERIALIZERS):
    if method == 'dataclass':
        return from_dict_threaded(json.loads(serialized))
    elif method == 'pickle':
        return pickle.loads(serialized)
    elif method == 'dill':
        import dill
        return dill.loads(serialized)

class HasGroup:
    _group: 'Group'


class DatasetBackedAttribute:
    compression: Optional[int]
    chunks: Optional[Tuple[int]]
    dtype: str
    shape: Tuple[int]
    maxshape: Optional[Tuple[int]]

    def __init__(self, dtype="|S512", compression=None, chunks=None, shape=(), maxshape=None):
        self.dtype = dtype
        self.compression = compression
        self.shape = shape
        self.chunks = chunks
        self.maxshape = maxshape

    def __set_name__(self, owner: HasGroup, name: str):
        self.public_name = name
        self.private_name = f"_{name}"

    def __get__(self, obj: HasGroup, objtype=None):
        # check cache:
        cached_val = getattr(obj, self.private_name, CACHE_MISS)
        if cached_val is CACHE_MISS:
            backing_val = self._deserialize(self.get_dataset(obj)[()], obj)
            setattr(obj, self.private_name, backing_val)
            cached_val = backing_val
        return cached_val

    def __set__(self, obj: HasGroup, value):
        dataset = self.get_dataset(obj)
        if self.maxshape is not None and value.shape != dataset.shape:
            # resizable dataset:
            dataset.resize(value.shape)
        dataset[()] = self._serialize(value, obj)
        dataset.flush()
        setattr(obj, self.private_name, value)

    def get_dataset(self, obj):
        return obj._group.require_dataset(self.public_name, shape=self.shape, dtype=self.dtype, compression=self.compression, maxshape=self.maxshape, exact='maxshape')

    def _serialize(self, value, obj=None):
        return value
    
    def _deserialize(self, value, obj=None):
        return value


class StringAttribute(DatasetBackedAttribute):
    def _serialize(self, value, obj=None):
        return value.encode()
    
    def _deserialize(self, value, obj=None):
        return value.decode()


class JSONAttribute(DatasetBackedAttribute):
    def _serialize(self, value, obj=None):
        return json.dumps(value)

    def _deserialize(self, value, obj=None):
        return json.loads(value) if value else None


class FitProblemAttribute(DatasetBackedAttribute):
    def _serialize(self, value, obj):
        return serialize_problem(value, obj.serializer)

    def _deserialize(self, value, obj):
        return deserialize_problem(value[0], obj.serializer)


class ProblemState(HasGroup):
    pathlist: Optional[List[str]] = JSONAttribute()
    fitProblem: Optional[bumps.fitproblem.FitProblem] = FitProblemAttribute(shape=(1,), dtype=f"|S{MAX_PROBLEM_SIZE}", compression=COMPRESSION)
    serializer: Optional[SERIALIZERS] = StringAttribute()
    filename: str = StringAttribute()

    def __init__(self, group: 'Group'):
        self._group = group
        # call the getters to initialize HDF backing:
        for attrname in ['filename', 'serializer', 'pathlist', 'fitProblem']:
            getattr(self, attrname)

UNCERTAINTY_DTYPE = 'f'
MAX_LABEL_LENGTH = 1024
LABEL_DTYPE = f"|S{MAX_LABEL_LENGTH}"

class UncertaintyState(HasGroup):
    AR = DatasetBackedAttribute(shape=(0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None,))
    gen_draws = DatasetBackedAttribute(shape=(0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None,))
    labels = DatasetBackedAttribute(shape=(0), compression=COMPRESSION, dtype=LABEL_DTYPE, maxshape=(None,))
    thin_draws = DatasetBackedAttribute(shape=(0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None,))
    gen_logp = DatasetBackedAttribute(shape=(0,0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None, None), chunks=(100,100))
    thin_logp = DatasetBackedAttribute(shape=(0,0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None, None), chunks=(100,100))
    thin_point = DatasetBackedAttribute(shape=(0,0,0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None, None, None), chunks=(100,100,100))
    update_CR_weight = DatasetBackedAttribute(shape=(0,0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None, None), chunks=(100,100))
    update_draws = DatasetBackedAttribute(shape=(0), compression=COMPRESSION, dtype=UNCERTAINTY_DTYPE, maxshape=(None,))

    def __init__(self, group: 'Group'):
        self._group = group
        # call the getters to initialize HDF backing:
        for attrname in ['AR', 'gen_draws', 'labels', 'thin_draws', 'gen_logp', 'thin_logp', 'thin_point', 'update_CR_weight', 'update_draws']:
            getattr(self, attrname)


class FittingState(HasGroup):
    # cache items:
    population: np.ndarray = DatasetBackedAttribute(dtype='f', compression=COMPRESSION, chunks=(10000,100), shape=(0,0), maxshape=(None, None))
    uncertainty_state: 'MCMCDraw'
    uncertainty_group: 'Group'
    _uncertainty_state_storage: UncertaintyState
    _uncertainty_state: 'MCMCDraw'

    def __init__(self, group: 'Group'):
        self._group = group
        self.uncertainty_group = group.require_group('uncertainty_state')
        self._uncertainty_state_storage = UncertaintyState(self.uncertainty_group)
        for attrname in ['population']:
            getattr(self, attrname)

    @property
    def uncertainty_state(self):
        cached_val = getattr(self, '_uncertainty_state', CACHE_MISS)
        if cached_val is CACHE_MISS:
            backing_val = read_uncertainty_state(self._uncertainty_state_storage)
            if backing_val is not None:
                setattr(self, '_uncertainty_state', backing_val)
            cached_val = backing_val
        return cached_val

    @uncertainty_state.setter
    def uncertainty_state(self, value: 'MCMCDraw'):
        write_uncertainty_state(value, self._uncertainty_state_storage)
        self._uncertainty_state = value


class Topic:
    dataset: 'Dataset'
    maxlen: Optional[int] = 1

    def __init__(self, dataset: 'Dataset', maxlen: Optional[int] = 1):
        self.dataset = dataset
        self.maxlen = maxlen

    def __getitem__(self, index: int):
        return json.loads(self.dataset[index])
    
    def __setitem__(self, index: int, value: Dict):
        self.dataset[index] = json.dumps(value)
        self.dataset.flush()

    def __len__(self):
        return len(self.dataset)

    def append(self, value: Dict):
        current_len: int = self.dataset.size
        if self.maxlen is None or current_len < self.maxlen:
            self.dataset.resize((current_len+1,))
            self[current_len] = value
        else:
            # not a true ring buffer: just overwrite last element when full
            # (only works for length-1 buffers, but that's all we have)
            self[current_len - 1] = value
        self.dataset.flush()

class TopicsDict:
    group: 'Group'
    lookup: Dict[str, Topic]

    def __init__(self, group: 'Group'):
        self.group = group
        self.lookup = {}
        for topic_name, maxlen in (
            ("log", None),
            ("update_parameters", 1),
            ("update_model", 1),
            ("model_loaded", 1),
            ("fit_active", 1),
            ("convergence_update", 1),
            ("uncertainty_update", 1),
            ("fitter_settings", 1),
            ("fitter_active", 1)
        ):
            if topic_name in group:
                topic_dataset = group[topic_name]
            else:
                topic_dataset = group.create_dataset(topic_name, shape=(0,), maxshape=(maxlen,), dtype=f"|S{MAX_TOPIC_MESSAGE}", compression=COMPRESSION)
            self.lookup[topic_name] = Topic(topic_dataset, maxlen=maxlen)

    def __getitem__(self, key: 'TopicNameType') -> Topic:
        return self.lookup[key]
    
    def get(self, key: 'TopicNameType', default: Any) -> Topic:
        return self.lookup[key] if key in self.lookup else default

    def items(self):
        return self.lookup.items()

class State:
    hostname: str
    port: int
    parallel: int
    problem: ProblemState
    fitting: FittingState
    topics: TopicsDict
    fit_thread: Optional['FitThread'] = None
    fit_abort: Optional[Event] = None
    fit_abort_event: Event
    fit_complete_event: Event
    calling_loop: Optional[asyncio.AbstractEventLoop] = None
    fit_enabled: Event
    session_file_name: Optional[str]

    _session_file: "File"

    def __init__(self, session_file_name: Optional[str] = None):
        # self.problem = problem
        # self.fitting = fitting if fitting is not None else FittingState()
        self.fit_abort_event = Event()
        self.fit_complete_event = Event()
        self.setup_backing(session_file_name)

    def __enter__(self):
        return self

    def setup_backing(self, session_file_name: Optional[str] = SESSION_FILE_NAME, read_only: bool = False ):
        import h5py
        backing_store = (session_file_name is not None)
        hdf_kw = dict(libver=HDF5_VERSION, driver="core", backing_store=backing_store)
        if read_only:
            hdf_kw["swmr"] = True
        mode = "r" if read_only else "a"
        self.session_file_name = session_file_name
        hdf_filename = session_file_name if session_file_name is not None else ":memory:"
        old_session = getattr(self, '_session_file', None)
        self._session_file = session_file = h5py.File(hdf_filename, mode, **hdf_kw)
        

        topics_group = session_file.require_group("topics")
        problem_group = session_file.require_group("problem")
        fitting_group = session_file.require_group("fitting")
        self.topics = TopicsDict(topics_group)
        self.problem = ProblemState(problem_group)
        self.fitting = FittingState(fitting_group)
        if not read_only:
            session_file.swmr_mode = True

        # close any open session files 
        if hasattr(old_session, 'close'):
            old_session.close()

    def copy_session_file(self, session_copy_name: str):
        import h5py
        if session_copy_name == self.session_file_name:
            warnings.warn(f"Can not save a copy with current filename: {session_copy_name}")
            return
        if self.session_file_name is not None:
            with h5py.File(self.session_file_name, "r", libver=HDF5_VERSION, swmr=True) as source, h5py.File(session_copy_name, "w", libver=HDF5_VERSION) as dest:
                for key in source:
                    source.copy(key, dest, name=key)
        else:
            source = self._session_file
            with h5py.File(session_copy_name, "w", libver=HDF5_VERSION) as dest:
                for key in source:
                    source.copy(key, dest, name=key)

    def cleanup(self):
        self._session_file.close()

    def __del__(self):
        self.cleanup()

    async def async_cleanup(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cleanup()



def write_uncertainty_state(state: 'MCMCDraw', storage: UncertaintyState):
        # Build 2-D data structures
        storage.gen_draws, storage.gen_logp = state.logp(full=True)
        _, storage.AR = state.acceptance_rate()

        storage.thin_draws, storage.thin_point, storage.thin_logp = state.chains()
        storage.update_draws, storage.update_CR_weight = state.CR_weight()
        storage.labels = np.array(state.labels, dtype=LABEL_DTYPE)

def read_uncertainty_state(loaded: UncertaintyState, skip=0, report=0, derived_vars=0):

    # Guess dimensions
    Ngen = loaded.gen_draws.shape[0]
    thinning = 1
    Nthin, Npop, Nvar = loaded.thin_point.shape
    Nupdate, Ncr = loaded.update_CR_weight.shape
    Nthin -= skip

    # Create empty draw and fill it with loaded data
    state = MCMCDraw(0, 0, 0, 0, 0, 0, thinning)
    #print("gen, var, pop", Ngen, Nvar, Npop)
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

    return state