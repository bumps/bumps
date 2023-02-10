from typing import TYPE_CHECKING, Optional, Dict, List, Any, Literal, cast
import json
from queue import Queue
from bumps.serialize import from_dict, to_dict
import h5py
import numpy as np

import refl1d.fitproblem
from bumps.dream.state import MCMCDraw


if TYPE_CHECKING:
    from .webserver import TopicNameType
    from h5py import Group, Dataset
    from .fit_thread import FitThread

# slow, small:
COMPRESSION = 9
MAX_TOPIC_MESSAGE = 1024*100 # 100k
MAX_PROBLEM_SIZE = 10*1024*1024 # 10 MB problem max size
SESSION_FILE_NAME = "session.h5"

def to_hdf5_group(state: 'State', group: 'Group'):
    import h5py
    fitting_grp = group.create_group("fitting")
    topics_grp = group.create_group("topics")
    problem_grp = group.create_group("problem")

    if state.fitting.population is not None:
        pop_shape = state.fitting.population.shape
        fitting_grp.create_dataset("population", shape=pop_shape, maxshape=(None, pop_shape[1]), compression = COMPRESSION)
    if state.fitting.uncertainty_state is not None:
        write_uncertainty_state(state.fitting.uncertainty_state, fitting_grp)

    for topic, contents in state.topics.items():
        num_entries = len(contents)
        maxlen = contents.maxlen
        topic_dataset = topics_grp.create_dataset(topic, shape=(num_entries,), maxshape=(contents.maxlen), dtype=f"|S{MAX_TOPIC_MESSAGE}", compression=COMPRESSION)
        for idx, message in enumerate(contents):
            topic_dataset[idx] = json.dumps(message)

    if state.problem is not None:
        if state.problem.fitProblem is not None:
            problem_json = json.dumps(to_dict(state.problem.fitProblem))
            problem_grp['fitProblem'] = problem_json
            problem_grp['fitProblem'].attrs["Content-Type"] = "application/json"
        if state.problem.filename is not None:
            problem_grp['filename'] = state.problem.filename
        if state.problem.pathlist is not None:
            pathlist_json = json.dumps(state.problem.pathlist)
            problem_grp['pathlist'] = pathlist_json
            problem_grp['pathlist'].attrs["Content-Type"] = "application/json"

CACHE_MISS = object()

class ProblemState:
    _group: h5py.Group
    _filename: Optional[str]
    _pathlist: Optional[List[str]]
    _fitProblem: Optional[refl1d.fitproblem.FitProblem]

    def __init__(self, group: h5py.Group):
        self._group = group

    @property
    def filename(self) -> Optional[str]:
        # check cache:
        cached_val = getattr(self, '_filename', CACHE_MISS)
        if cached_val is CACHE_MISS:
            backing_val = self._group['filename'][()] if 'filename' in self._group else None
            setattr(self, '_filename', backing_val)
            cached_val = backing_val
        return cached_val
    
    @filename.setter
    def filename(self, value: str):
        dset = self._group.require_dataset('filename', (), h5py.vlen_dtype(str))
        dset[()] = value
        self._filename = value

    @property
    def pathlist(self) -> Optional[List[str]]:
        cached_val = getattr(self, '_pathlist', CACHE_MISS)
        if cached_val is CACHE_MISS:
            backing_val = json.loads(self._group['pathlist'][()]) if 'pathlist' in self._group else None
            setattr(self, '_pathlist', backing_val)
            cached_val = backing_val
        return cached_val
    
    @pathlist.setter
    def pathlist(self, value: List[str]):
        dset = self._group.require_dataset('pathlist', (), h5py.vlen_dtype(str))
        dset[()] = json.dumps(value)
        self._pathlist = value

    @property
    def fitProblem(self) -> Optional[refl1d.fitproblem.FitProblem]:
        cached_val = getattr(self, '_fitProblem', CACHE_MISS)
        if cached_val is CACHE_MISS:
            backing_val = from_dict(json.loads(self._group['fitProblem'][()])) if 'fitProblem' in self._group else None
            setattr(self, '_fitProblem', backing_val)
            cached_val = backing_val
        return cached_val
    
    @fitProblem.setter
    def fitProblem(self, value: refl1d.fitproblem.FitProblem):
        dset = self._group.require_dataset('fitProblem', (), h5py.vlen_dtype(str))
        dset[()] = json.dumps(to_dict(value))
        self._fitProblem = value

class FittingState:
    _group: h5py.Group
    # cache items:
    _population: np.ndarray
    _uncertainty_state: 'MCMCDraw'

    def __init__(self, group: h5py.Group):
        self._group = group

    @property
    def population(self) -> Optional[np.ndarray]:
        cached_val = getattr(self, '_population', CACHE_MISS)
        if cached_val is CACHE_MISS:
            backing_val = self._group['population'][()] if 'population' in self._group else None
            if backing_val is not None:
                setattr(self, '_population', backing_val)
            cached_val = backing_val
        return cached_val
    
    @population.setter
    def population(self, value: np.ndarray):
        pop_shape = value.shape
        if not 'population' in self._group:
            self._group.create_dataset("population", shape=pop_shape, data=value, maxshape=(None, None), compression = COMPRESSION)
        else:
            dset = cast(h5py.Dataset, self._group["population"])
            if dset.shape != pop_shape:
                dset.resize(pop_shape)        
            dset[()] = value
        self._population = value

    @property
    def uncertainty_state(self):
        cached_val = getattr(self, '_uncertainty_state', CACHE_MISS)
        if cached_val is CACHE_MISS:
            backing_val = read_uncertainty_state(self._group) if 'gen_draws' in self._group else None
            if backing_val is not None:
                setattr(self, '_uncertainty_state', backing_val)
            cached_val = backing_val
        return cached_val

    @uncertainty_state.setter
    def uncertainty_state(self, value: 'MCMCDraw'):
        print("writing uncertainty state: ", value)
        write_uncertainty_state(value, self._group)
        self._uncertainty_state = value
        print("done writing uncertainty state.")

class Topic:
    dataset: h5py.Dataset
    maxlen: Optional[int] = 1

    def __init__(self, dataset: h5py.Dataset, maxlen: Optional[int] = 1):
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

    def __init__(self, group: 'Group'):
        self.group = group
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
    
    def __getitem__(self, key: 'TopicNameType') -> Topic:
        topic_dataset = cast(h5py.Dataset, self.group[key])
        return Topic(topic_dataset, maxlen=topic_dataset.maxshape[0])
    
    def get(self, key: 'TopicNameType', default: Any) -> Topic | Any:
        return self[key] if key in self.group else default

    def items(self):
        return [(topic_name, self[topic_name]) for topic_name in self.group.keys()]

class State:
    hostname: str
    port: int
    problem: ProblemState
    dream: FittingState
    topics: TopicsDict
    fit_thread: Optional['FitThread'] = None
    abort_queue: Queue

    def __init__(self, session_file_name: str = SESSION_FILE_NAME):
        # self.problem = problem
        # self.fitting = fitting if fitting is not None else FittingState()
        self.abort_queue = Queue()
        import h5py
        self.session_file = session_file = h5py.File(session_file_name, "a", libver='latest')
        topics_group = session_file.require_group("topics")
        problem_group = session_file.require_group("problem")
        fitting_group = session_file.require_group("fitting")
        self.topics = TopicsDict(topics_group)
        self.problem = ProblemState(problem_group)
        self.fitting = FittingState(fitting_group)
        session_file.swmr_mode = True

    async def cleanup(self):
        self.session_file.close()

def write_uncertainty_state(state: 'MCMCDraw', group: 'Group', compression=COMPRESSION):
        to_save = {}
        # Build 2-D data structures
        to_save["gen_draws"], to_save["gen_logp"] = state.logp(full=True)
        _, to_save["AR"] = state.acceptance_rate()

        to_save["thin_draws"], to_save["thin_point"], to_save["thin_logp"] = state.chains()
        to_save["update_draws"], to_save["update_CR_weight"] = state.CR_weight()

        #TODO: missing _outliers from save_state
        for field_name, data in to_save.items():
            if field_name in group:
                dataset = cast(h5py.Dataset, group[field_name])
                if dataset.shape != data.shape:
                    dataset.resize(data.shape)
                dataset[()] = data
            else:
                maxshape = tuple([None for dim in data.shape])
                dataset = group.create_dataset(field_name, data=data, maxshape=maxshape, compression=compression)


def read_uncertainty_state(group: 'Group', skip=0, report=0, derived_vars=0):

    loaded: Dict[str, np.ndarray] = dict(group.items())
    
    # Guess dimensions
    Ngen = loaded["gen_draws"].shape[0]
    thinning = 1
    Nthin, Npop, Nvar = loaded["thin_point"].shape
    Nupdate, Ncr = loaded["update_CR_weight"].shape
    Nthin -= skip

    # Create empty draw and fill it with loaded data
    state = MCMCDraw(0, 0, 0, 0, 0, 0, thinning)
    #print("gen, var, pop", Ngen, Nvar, Npop)
    state.draws = Ngen * Npop
    state.generation = Ngen
    state._gen_index = 0
    state._gen_draws = loaded["gen_draws"][()]
    state._gen_acceptance_rate = loaded["AR"][()]
    state._gen_logp = loaded["gen_logp"][()]
    state.thinning = thinning
    state._thin_count = Ngen//thinning
    state._thin_index = 0
    state._thin_draws = loaded["thin_draws"][()]
    state._thin_logp = loaded["thin_logp"][()]
    state._thin_point = loaded["thin_point"][()]
    state._gen_current = state._thin_point[-1].copy()
    state._update_count = Nupdate
    state._update_index = 0
    state._update_draws = loaded["update_draws"][()]
    state._update_CR_weight = loaded["update_CR_weight"][()]
    state._outliers = []

    bestidx = np.unravel_index(np.argmax(loaded["thin_logp"]), loaded["thin_logp"].shape)
    state._best_logp = loaded["thin_logp"][bestidx]
    state._best_x = loaded["thin_point"][bestidx]
    state._best_gen = 0

    return state