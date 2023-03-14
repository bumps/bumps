from collections import deque
from queue import Queue
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import bumps, bumps.fitproblem, bumps.dream.state
    from .webserver import TopicNameType
    from .fit_thread import FitThread

class ProblemState:
    fitProblem: Optional['bumps.fitproblem.FitProblem'] = None
    pathlist: Optional[List[str]] = None
    filename: Optional[str] = None

class FittingState:
    abort: bool = False
    population: Optional[List] = None
    uncertainty_state: Optional['bumps.dream.state.MCMCDraw']


class State:
    # These attributes are ephemeral, not to be serialized/stored:
    hostname: str
    port: int
    abort_queue: Queue
    fit_thread: Optional['FitThread'] = None

    # State to be stored:
    problem: ProblemState
    fitting: FittingState
    topics: Dict['TopicNameType', 'deque[Dict]']

    def __init__(self, problem: Optional[ProblemState] = None, fitting: Optional[FittingState] = None):
        self.problem = problem if problem is not None else ProblemState()
        self.fitting = FittingState()
        self.abort_queue = Queue()
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

    async def cleanup(self):
        pass