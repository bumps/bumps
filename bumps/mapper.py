"""
Parallel and serial mapper implementations.

The API is a bit crufty since interprocess communication has evolved from
the original implementation. And the names are misleading.

Usage::

    Mapper.start_worker(problem)
    mapper = Mapper.start_mapper(problem, None, cpus)
    result = mapper(points)
    ...
    mapper = Mapper.start_mapper(problem, None, cpus)
    result = mapper(points)
    Mapper.stop_mapper()
"""

import sys
import os
import signal

# {{{ http://code.activestate.com/recipes/496767/ (r1)
# Converted to use ctypes by Paul Kienzle


PROCESS_ALL_ACCESS = 0x1F0FFF


def can_pickle(problem, check=False):
    """
    Returns True if *problem* can be pickled.

    If this method returns False then MPMapper cannot be used and
    SerialMapper should be used instead.

    If *check* is True then call *nllf()* on the duplicated object as a
    "smoke test" to verify that the function will run after copying. This
    is not foolproof. For example, access to a database may work in the
    duplicated object because the connection is open and available in the
    current process, but it will fail when trying to run on a remote machine.
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


def setpriority(pid=None, priority=1):
    """
    Set The Priority of a Windows Process.  Priority is a value between 0-5
    where 2 is normal priority and 5 is maximum.  Default sets the priority
    of the current python process but can take any valid process ID.
    """

    # import win32api,win32process,win32con
    from ctypes import windll

    priorityclasses = [
        0x40,  # IDLE_PRIORITY_CLASS,
        0x4000,  # BELOW_NORMAL_PRIORITY_CLASS,
        0x20,  # NORMAL_PRIORITY_CLASS,
        0x8000,  # ABOVE_NORMAL_PRIORITY_CLASS,
        0x80,  # HIGH_PRIORITY_CLASS,
        0x100,  # REALTIME_PRIORITY_CLASS
    ]
    if pid is None:
        pid = windll.kernel32.GetCurrentProcessId()
    handle = windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, True, pid)
    windll.kernel32.SetPriorityClass(handle, priorityclasses[priority])


# end of http://code.activestate.com/recipes/496767/ }}}


def nice():
    if os.name == "nt":
        setpriority(priority=1)
    else:
        os.nice(5)


# Noise so that the type checker is happy
class BaseMapper(object):
    has_problem = False

    @staticmethod
    def start_worker(problem):
        """Called with the problem to initialize the worker"""
        raise NotImplementedError()

    @staticmethod
    def start_mapper(problem, modelargs=None, cpus=0):
        """Called with the problem on a new fit."""
        raise NotImplementedError()

    # TODO: deprecate mapper parameter
    @staticmethod
    def stop_mapper(mapper=None):
        raise NotImplementedError()


class SerialMapper(BaseMapper):
    @staticmethod
    def start_worker(problem):
        pass

    @staticmethod
    def start_mapper(problem, modelargs=None, cpus=0):
        # Note: map is n iterator in python 3.x
        return lambda points: list(map(problem.nllf, points))

    @staticmethod
    def stop_mapper(mapper=None):
        pass


# Load the problem in the remote process rather than pickling
# def _MP_load_problem(*modelargs):
#    from .fitproblem import load_problem
#    _MP_set_problem(load_problem(*modelargs))


def _MP_setup():
    # Using MPMapper class variables to store worker globals.
    # It doesn't matter if they conflict with the controller values since
    # they are in a different process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    nice()


def _MP_run_problem(problem_point_tuple):
    problem_id, point, shared_pickled_problem = problem_point_tuple
    if problem_id != MPMapper.problem_id:
        # print(f"Fetching problem {problem_id} from namespace")
        # Problem is pickled using dill when it is available
        try:
            import dill

            MPMapper.problem = dill.loads(shared_pickled_problem[:].tobytes())
        except ImportError:
            import pickle

            MPMapper.problem = pickle.loads(shared_pickled_problem[:].tobytes())
        MPMapper.problem_id = problem_id
    return MPMapper.problem.nllf(point)


class MPMapper(BaseMapper):
    # Note: suprocesses are using the same variables
    pool = None
    manager = None
    problem_id = 0
    shared_pickled_problem = None
    problem = None

    @staticmethod
    def start_worker(problem):
        pass

    @staticmethod
    def start_mapper(problem, modelargs=None, cpus=0):
        import multiprocessing

        # Set up the process pool on the first call.
        if MPMapper.pool is None:
            # Create a sync namespace to distribute the problem description.
            MPMapper.manager = multiprocessing.Manager()
            # Start the process pool, sending the namespace handle
            if cpus == 0:
                cpus = multiprocessing.cpu_count()
            MPMapper.pool = multiprocessing.Pool(cpus, _MP_setup)

        # Increment the problem number and store the problem in the namespace.
        # The store action uses pickle to transfer python objects to the
        # manager process. Since this may fail for lambdas and for functions
        # defined within the model file, instead use dill (if available)
        # to pickle the problem before storing.
        MPMapper.problem_id += 1
        try:
            import dill

            MPMapper.pickled_problem = dill.dumps(problem, recurse=True)
        except ImportError:
            import pickle

            MPMapper.pickled_problem = pickle.dumps(problem)
        MPMapper.shared_pickled_problem = MPMapper.manager.Array("B", MPMapper.pickled_problem)

        # Set the mapper to send problem_id/point/shared_pickled_problem value triples
        def mapper(points):
            try:
                return MPMapper.pool.map(
                    _MP_run_problem, ((MPMapper.problem_id, p, MPMapper.shared_pickled_problem) for p in points)
                )
            except KeyboardInterrupt:
                MPMapper.stop_mapper()

        return mapper

    @staticmethod
    def stop_mapper(mapper=None):
        # reset pool and manager
        if MPMapper.pool is not None:
            MPMapper.pool.terminate()
            MPMapper.pool = None
            MPMapper.manager.shutdown()
            MPMapper.manager = None
        # Don't reset problem id; it keeps count even when mapper is restarted.
        ##MPMapper.problem_id = 0


def _MPI_set_problem(problem, comm, root=0):
    import dill

    pickled_problem = dill.dumps(problem, recurse=True) if comm.rank == root else None
    pickled_problem = comm.bcast(pickled_problem, root=root)
    return problem if comm.rank == root else dill.loads(pickled_problem)


def _MPI_map(problem, points, comm, root=0):
    import numpy as np
    from mpi4py import MPI

    # print(f"{comm.rank}: mapping points")

    # Send number of points and number of variables per point.
    # root: return result if there are points otherwise return False
    # worker: return True if there are points otherwise return False
    npoints, nvars = comm.bcast(points.shape if comm.rank == root else None, root=root)
    if npoints == 0:
        return None

    # Divvy points equally across all processes
    whole = points if comm.rank == root else None
    idx = np.arange(comm.size)
    size = np.ones(comm.size, idx.dtype) * (npoints // comm.size) + (idx < npoints % comm.size)
    offset = np.cumsum(np.hstack((0, size[:-1])))
    part = np.empty((size[comm.rank], nvars), dtype="d")
    comm.Scatterv((whole, (size * nvars, offset * nvars), MPI.DOUBLE), (part, MPI.DOUBLE), root=root)

    # Evaluate models assigned to each processor
    partial_result = np.array([problem.nllf(pk) for pk in part], dtype="d")

    # Collect results
    result = np.empty(npoints, dtype="d") if comm.rank == root else True
    comm.Barrier()
    comm.Gatherv((partial_result, MPI.DOUBLE), (result, (size, offset), MPI.DOUBLE), root=root)
    comm.Barrier()
    return result


def using_mpi():
    # Can look for environment variables defined by mpirun
    #   mpich: PMI_RANK, PMI_*, MPI_*
    #   openmp: OMPI_COMM_WORLD_RANK, OMPI_* PMIX_*
    #   impi_rt (intel): PMI_RANK I_MPI_* HYDRA_*
    #   msmpi (microsoft): PMI_RANK PMI_* MSMPI_*
    # TODO: Make sure that Slurm isn't setting MPI variables in the batch script.
    # I seem to recall some messiness with pmix variants on our local openmpi cluster
    # so I would rather look at OMP_COMM_WORLD_RANK
    mpienv = [
        "OMPI_COMM_WORLD_RANK",  # OpenMPI
        "PMI_RANK",  # MPICH, MSMPI [Microsoft], IMPI_RT [Intel]
    ]
    return any(v in os.environ for v in mpienv)

    # The robust solution is as follows, but it requires opening the MPI ports.
    # This triggers a security box on the Mac asking to give the python interpreter
    # access to these ports. Since MPI use on Mac will be rare (only for problems
    # that can't be pickled) we should avoid the confusing security box.
    # from mpi4py import MPI
    # try:
    #     comm = MPI.COMM_WORLD
    #     return comm.size > 1
    # finally:
    #     return False


class MPIMapper(BaseMapper):
    has_problem = True
    """For MPIMapper only the worker is initialized with the fit problem."""

    @staticmethod
    def start_worker(problem):
        """
        Start the worker process.

        For the main process this does nothing and returns immediately. The
        worker processes never return.

        Each worker sits in a loop waiting for the next batch of points
        for the problem, or for the next problem. Set t
        problem is set to None, then exit the process and never
        """
        from mpi4py import MPI

        comm, root = MPI.COMM_WORLD, 0
        MPIMapper.rank = comm.rank
        rank = comm.rank
        # print(f"MPI {rank} of {comm.size} initializing")

        # If worker, sit in a loop waiting for the next point.
        # If the point is empty, then wait for a new problem.
        # If the problem is None then we are done, otherwise wait for next point.
        if rank != root:
            # print(f"{rank}: looping")
            while True:
                result = _MPI_map(problem, None, comm, root)
                if result is None:
                    problem = _MPI_set_problem(None, comm, root)
                    if problem is None:
                        break
                    # print(f"{rank}: changing problem")

            # print(f"{rank}: finalizing")
            MPI.Finalize()

            # Exit the program after the worker is done. Don't return
            # to the caller since that is continuing on with the main
            # thread, and in particular, attempting to rerun the fit on
            # each worker.
            # print(f"{rank}: Worker exiting")
            sys.exit(0)
            # print(f"{rank}: Worker exited")
        else:
            # Root initialization:
            MPIMapper.has_problem = problem is not None
            # print("mapper has problem", MPIMapper.has_problem)

    @staticmethod
    def start_mapper(problem, modelargs=None, cpus=0):
        # Only root can get here---worker is stuck in start_worker
        from mpi4py import MPI
        import numpy as np

        comm, root = MPI.COMM_WORLD, 0

        # Signal new problem then send it, but not on the first fit. We do this
        # so that we can still run MPI fits even if the problem itself cannot
        # be pickled, but only the first one. (You can still fit a series even
        # if the problem can't be pickled, but you will need to restart the
        # MPI job separately for each fit.)
        # Note: setting problem to None stops the program, so call finalize().
        def mapper(points):
            return _MPI_map(problem, points, comm, root)

        if not MPIMapper.has_problem:  # Only true on the first fit
            # print(f"*** {comm.rank}: replacing problem")
            # Send an empty set of points to signal a new problem is coming.
            mapper(np.empty((0, 0), "d"))
            _MPI_set_problem(problem, comm, root)
            if problem is None:
                # print(f"{comm.rank}: finalizing root")
                MPI.Finalize()
        MPIMapper.has_problem = False
        return mapper

    @staticmethod
    def stop_mapper(mapper=None):
        # print("stopping mapper")
        # Set problem=None to stop the program.
        MPIMapper.start_mapper(None, None)
