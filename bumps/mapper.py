"""
Parallel and serial mapper implementations.
"""
import sys
import os

# {{{ http://code.activestate.com/recipes/496767/ (r1)
# Converted to use ctypes by Paul Kienzle


PROCESS_ALL_ACCESS = 0x1F0FFF


def setpriority(pid=None, priority=1):
    """
    Set The Priority of a Windows Process.  Priority is a value between 0-5
    where 2 is normal priority and 5 is maximum.  Default sets the priority
    of the current python process but can take any valid process ID.
    """

    #import win32api,win32process,win32con
    from ctypes import windll

    priorityclasses = [0x40,   # IDLE_PRIORITY_CLASS,
                       0x4000,  # BELOW_NORMAL_PRIORITY_CLASS,
                       0x20,   # NORMAL_PRIORITY_CLASS,
                       0x8000,  # ABOVE_NORMAL_PRIORITY_CLASS,
                       0x80,   # HIGH_PRIORITY_CLASS,
                       0x100,  # REALTIME_PRIORITY_CLASS
                      ]
    if pid is None:
        pid = windll.kernel32.GetCurrentProcessId()
    handle = windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, True, pid)
    windll.kernel32.SetPriorityClass(handle, priorityclasses[priority])
# end of http://code.activestate.com/recipes/496767/ }}}


def nice():
    if os.name == 'nt':
        setpriority(priority=1)
    else:
        os.nice(5)


class SerialMapper(object):

    @staticmethod
    def start_worker(problem):
        pass

    @staticmethod
    def start_mapper(problem, modelargs, cpus=0):
        # Note: map is n iterator in python 3.x
        return lambda points: list(map(problem.nllf, points))

    @staticmethod
    def stop_mapper(mapper):
        pass


# Load the problem in the remote process rather than pickling
#def _MP_load_problem(*modelargs):
#    from .fitproblem import load_problem
#    _MP_set_problem(load_problem(*modelargs))

def _MP_setup(namespace):
    # Using MPMapper class variables to store worker globals.
    # It doesn't matter if they conflict with the controller values since
    # they are in a different process.
    MPMapper.namespace = namespace
    nice()


def _MP_run_problem(problem_point_pair):
    problem_id, point = problem_point_pair
    if problem_id != MPMapper.problem_id:
        #print(f"Fetching problem {problem_id} from namespace")
        # Problem is pickled using dill when it is available
        try:
            import dill
            MPMapper.problem = dill.loads(MPMapper.namespace.pickled_problem)
        except ImportError:
            MPMapper.problem = MPMapper.namespace.problem
        MPMapper.problem_id = problem_id
    return MPMapper.problem.nllf(point)


class MPMapper(object):
    # Note: suprocesses are using the same variables
    pool = None
    manager = None
    namespace = None
    problem_id = 0

    @staticmethod
    def can_pickle(problem, check=False):
        """
        Returns True if *problem* can be pickled.

        If this method returns False then MPMapper cannot be used and
        SerialMapper should be used instead.

        If *check* is True then call *nllf()* on the duplicated object. This
        will not be a foolproof check. If the model uses ephemeral objects,
        such as a handle to an external process or similar, then handle might
        be copied and accessible locally but not be accessible to the remote
        process.
        """
        try:
            import dill
        except ImportError:
            dill = None
            import pickle
        try:
            if dill is not None:
                dup = dill.loads(dill.dumps(problem, recurse=True))
            else:
                dup = pickle.loads(pickle.dumps(problem))
            if check:
                dup.nllf()
            return True
        except Exception:
            return False

    @staticmethod
    def start_worker(problem):
        pass

    @staticmethod
    def start_mapper(problem, modelargs, cpus=0):
        import multiprocessing

        # Set up the process pool on the first call.
        if MPMapper.pool is None:
            # Create a sync namespace to distribute the problem description.
            MPMapper.manager = multiprocessing.Manager()
            MPMapper.namespace = MPMapper.manager.Namespace()
            # Start the process pool, sending the namespace handle
            if cpus == 0:
                cpus = multiprocessing.cpu_count()
            MPMapper.pool = multiprocessing.Pool(cpus, _MP_setup, (MPMapper.namespace,))

        # Increment the problem number and store the problem in the namespace.
        # The store action uses pickle to transfer python objects to the
        # manager process. Since this may fail for lambdas and for functions
        # defined within the model file, instead use dill (if available)
        # to pickle the problem before storing.
        MPMapper.problem_id += 1
        try:
            import dill
            MPMapper.namespace.pickled_problem = dill.dumps(problem, recurse=True)
        except ImportError:
            MPMapper.namespace.problem = problem
        ## Store the modelargs and the problem name if pickling doesn't work
        #MPMapper.namespace.modelargs = modelargs

        # Set the mapper to send problem_id/point value pairs
        mapper = lambda points: MPMapper.pool.map(
            _MP_run_problem, ((MPMapper.problem_id, p) for p in points))
        return mapper

    @staticmethod
    def stop_mapper(mapper):
        MPMapper.pool.terminate()

def _MPI_set_problem(comm, problem, root=0):
    global _problem
    _problem = comm.bcast(problem)


def _MPI_run_problem(point):
    global _problem
    return _problem.nllf(point)


def _MPI_map(comm, points, root=0):
    import numpy as np
    from mpi4py import MPI
    # Send number of points and number of variables per point
    npoints, nvars = comm.bcast(
        points.shape if comm.rank == root else None, root=root)
    if npoints == 0:
        raise StopIteration

    # Divvy points equally across all processes
    whole = points if comm.rank == root else None
    idx = np.arange(comm.size)
    size = np.ones(comm.size, idx.dtype) * \
        (npoints // comm.size) + (idx < npoints % comm.size)
    offset = np.cumsum(np.hstack((0, size[:-1])))
    part = np.empty((size[comm.rank], nvars), dtype='d')
    comm.Scatterv((whole, (size * nvars, offset * nvars), MPI.DOUBLE),
                  (part, MPI.DOUBLE),
                  root=root)

    # Evaluate models assigned to each processor
    partial_result = np.array([_MPI_run_problem(pi) for pi in part],
                               dtype='d')

    # Collect results
    result = np.empty(npoints, dtype='d') if comm.rank == root else None
    comm.Barrier()
    comm.Gatherv((partial_result, MPI.DOUBLE),
                 (result, (size, offset), MPI.DOUBLE),
                 root=root)
    comm.Barrier()
    return result


class MPIMapper(object):

    @staticmethod
    def start_worker(problem):
        global _problem
        _problem = problem
        from mpi4py import MPI
        root = 0
        # If master, then return to main program
        if MPI.COMM_WORLD.rank == root:
            return
        # If slave, then set problem and wait in map loop
        #_MPI_set_problem(MPI.COMM_WORLD, None, root=root)
        try:
            while True:
                _MPI_map(MPI.COMM_WORLD, None, root=root)
        except StopIteration:
            pass
        MPI.Finalize()
        sys.exit(0)

    @staticmethod
    def start_mapper(problem, modelargs, cpus=0):
        # Slave started from start_worker, so it never gets here
        # Slave expects _MPI_set_problem followed by a series
        # of map requests
        from mpi4py import MPI
        #_MPI_set_problem(MPI.COMM_WORLD, problem)
        return lambda points: _MPI_map(MPI.COMM_WORLD, points)

    @staticmethod
    def stop_mapper(mapper):
        from mpi4py import MPI
        import numpy as np
        # Send an empty point list to stop the iteration
        try:
            mapper(np.empty((0, 0), 'd'))
            raise RuntimeException("expected StopIteration")
        except StopIteration:
            pass
        MPI.Finalize()


class AMQPMapper(object):

    @staticmethod
    def start_worker(problem):
        #sys.stderr = open("bumps-%d.log"%os.getpid(),"w")
        #print >>sys.stderr,"worker is starting"; sys.stdout.flush()
        from amqp_map.config import SERVICE_HOST
        from amqp_map.core import connect, start_worker as serve
        server = connect(SERVICE_HOST)
        #os.system("echo 'serving' > /tmp/map.%d"%(os.getpid()))
        # print "worker is serving"; sys.stdout.flush()
        serve(server, "bumps", problem.nllf)
        #print >>sys.stderr,"worker ended"; sys.stdout.flush()

    @staticmethod
    def start_mapper(problem, modelargs, cpus=0):
        import sys
        import multiprocessing
        import subprocess
        from amqp_map.config import SERVICE_HOST
        from amqp_map.core import connect, Mapper

        server = connect(SERVICE_HOST)
        mapper = Mapper(server, "bumps")
        cpus = multiprocessing.cpu_count()
        pipes = []
        for _ in range(cpus):
            cmd = [sys.argv[0], "--worker"] + modelargs
            # print "starting",sys.argv[0],"in",os.getcwd(),"with",cmd
            pipe = subprocess.Popen(cmd, universal_newlines=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pipes.append(pipe)
        for pipe in pipes:
            if pipe.poll() > 0:
                raise RuntimeError("subprocess returned %d\nout: %s\nerr: %s"
                    % (pipe.returncode, pipe.stdout, pipe.stderr))
        #os.system(" ".join(cmd+["&"]))
        import atexit

        def exit_fun():
            for p in pipes:
                p.terminate()
        atexit.register(exit_fun)

        # print "returning mapper",mapper
        return mapper

    @staticmethod
    def stop_mapper(mapper):
        for pipe in mapper.pipes:
            pipe.terminate()
