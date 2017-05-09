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


def _MP_set_problem(problem):
    global _problem
    nice()
    _problem = problem


def _MP_run_problem(point):
    global _problem
    return _problem.nllf(point)


class MPMapper(object):
    pool = None

    @staticmethod
    def start_worker(problem):
        pass

    @staticmethod
    def start_mapper(problem, modelargs, cpus=0):
        import multiprocessing
        if cpus==0:
            cpus = multiprocessing.cpu_count()
        if MPMapper.pool is not None:
            MPMapper.pool.terminate()
        #MPMapper.pool = multiprocessing.Pool(cpus,_MP_load_problem,modelargs)
        MPMapper.pool = multiprocessing.Pool(cpus, _MP_set_problem, (problem,))
        mapper = lambda points: MPMapper.pool.map(_MP_run_problem, points)
        return mapper

    @staticmethod
    def stop_mapper(mapper):
        pass


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
