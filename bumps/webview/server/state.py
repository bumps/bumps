from collections import deque
from queue import Queue
from typing import Dict, List, Optional, TYPE_CHECKING
from threading import Event
import asyncio

from .state_hdf5_backed import SERIALIZERS

if TYPE_CHECKING:
    import bumps, bumps.fitproblem, bumps.dream.state
    from .webserver import TopicNameType
    from .fit_thread import FitThread


SESSION_FILE_NAME = "session.h5"


class ProblemState:
    fitProblem: Optional["bumps.fitproblem.FitProblem"] = None
    pathlist: Optional[List[str]] = None
    serializer: Optional[SERIALIZERS] = None
    filename: Optional[str] = None


class FittingState:
    abort: bool = False
    population: Optional[List] = None
    uncertainty_state: Optional["bumps.dream.state.MCMCDraw"]


class State:
    # These attributes are ephemeral, not to be serialized/stored:
    hostname: str
    port: int
    parallel: int
    abort_queue: Queue
    fit_thread: Optional["FitThread"] = None
    fit_abort: Optional[Event] = None
    fit_abort_event: Event
    fit_complete_event: Event
    calling_loop: Optional[asyncio.AbstractEventLoop] = None
    fit_enabled: Event
    session_file_name: Optional[str]

    # State to be stored:
    problem: ProblemState
    fitting: FittingState
    topics: Dict["TopicNameType", "deque[Dict]"]

    def __init__(self, problem: Optional[ProblemState] = None, fitting: Optional[FittingState] = None):
        self.problem = problem if problem is not None else ProblemState()
        self.fitting = FittingState()
        self.fit_abort_event = Event()
        self.fit_complete_event = Event()
        self.topics = {
            "log": deque([]),
            "update_parameters": deque([], maxlen=1),
            "update_model": deque([], maxlen=1),
            "model_loaded": deque([], maxlen=1),
            "fit_active": deque([], maxlen=1),
            "convergence_update": deque([], maxlen=1),
            "uncertainty_update": deque([], maxlen=1),
            "fitter_settings": deque([], maxlen=1),
            "fitter_active": deque([], maxlen=1),
        }

    def setup_backing(self, session_file_name: Optional[str] = SESSION_FILE_NAME, read_only: bool = False):
        self.session_file_name = session_file_name

    def save(self):
        pass

    def copy_session_file(self, session_copy_name: str):
        pass

    def write_session_file(self):
        pass

    def cleanup(self):
        self._session_file.close()

    def __del__(self):
        self.cleanup()

    async def async_cleanup(self):
        self.cleanup()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cleanup()

    async def cleanup(self):
        pass
