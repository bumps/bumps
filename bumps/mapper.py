import os
from copy import deepcopy

## {{{ http://code.activestate.com/recipes/496767/ (r1)
## Converted to use ctypes by Paul Kienzle
def setpriority(pid=None,priority=1):
    """
    Set The Priority of a Windows Process.  Priority is a value between 0-5
    where 2 is normal priority and 5 is maximum.  Default sets the priority
    of the current python process but can take any valid process ID.
    """

    #import win32api,win32process,win32con
    from ctypes import windll

    priorityclasses = [0x40,   # IDLE_PRIORITY_CLASS,
                       0x4000, # BELOW_NORMAL_PRIORITY_CLASS,
                       0x20,   # NORMAL_PRIORITY_CLASS,
                       0x8000, # ABOVE_NORMAL_PRIORITY_CLASS,
                       0x80,   # HIGH_PRIORITY_CLASS,
                       0x100,  # REALTIME_PRIORITY_CLASS
                       ]
    if pid == None:
        pid = windll.kernel32.GetCurrentProcessId()
    PROCESS_ALL_ACCESS = 0x1F0FFF
    handle = windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, True, pid)
    windll.kernel32.SetPriorityClass(handle, priorityclasses[priority])
## end of http://code.activestate.com/recipes/496767/ }}}
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
    def start_mapper(problem, modelargs):
        return lambda points: map(problem.nllf, points)
    @staticmethod
    def stop_mapper(mapper):
        pass

def _MP_set_problem(problem):
    global _problem
    nice()
    _problem = problem
def _MP_run_problem(point):
    global _problem
    return _problem.nllf(point)
class MPMapper(object):
    @staticmethod
    def start_worker(problem):
        pass
    @staticmethod
    def start_mapper(problem, modelargs, cpus=None):
        import multiprocessing
        if cpus is None:
            cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus,_MP_set_problem,(problem,))
        mapper = lambda points: pool.map(_MP_run_problem, points)
        return mapper
    @staticmethod
    def stop_mapper(mapper):
        pass

class AMQPMapper(object):

    @staticmethod
    def start_worker(problem):
        #sys.stderr = open("bumps-%d.log"%os.getpid(),"w")
        #print >>sys.stderr,"worker is starting"; sys.stdout.flush()
        from .amqp_map.config import SERVICE_HOST
        from .amqp_map.core import connect, start_worker as serve
        server = connect(SERVICE_HOST)
        #os.system("echo 'serving' > /tmp/map.%d"%(os.getpid()))
        #print "worker is serving"; sys.stdout.flush()
        serve(server, "bumps", problem.nllf)
        #print >>sys.stderr,"worker ended"; sys.stdout.flush()

    @staticmethod
    def start_mapper(problem, modelargs):
        import multiprocessing
        from .amqp_map.config import SERVICE_HOST
        from .amqp_map.core import connect, Mapper

        server = connect(SERVICE_HOST)
        mapper = Mapper(server, "bumps")
        cpus = multiprocessing.cpu_count()
        pipes = []
        for _ in range(cpus):
            cmd = [sys.argv[0], "--worker"] + modelargs
            #print "starting",sys.argv[0],"in",os.getcwd(),"with",cmd
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
            for p in pipes: p.terminate()
        atexit.register(exit_fun)

        #print "returning mapper",mapper
        return mapper

    @staticmethod
    def stop_mapper(mapper):
        for pipe in mapper.pipes:
            pipe.terminate()
