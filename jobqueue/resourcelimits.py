"""
Resource limit handling.
"""

try:
    from signal import signal, SIGXCPU, SIG_IGN
    from resource import setrlimit, RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_DATA
except:
    print "limits not supported"
    RLIMIT_CPU=RLIMIT_FSIZE=RLIMIT_DATA=0
    SIGXCPU = SIG_IGN = 0
    def setrlimit(resource,limits): pass
    def signal(value, handler): pass

class CPULimit(Exception): pass

def _xcpuhandler(signum, frame):
    signal(SIGXCPU, SIG_IGN)
    raise CPULimit("CPU time exceeded.")

def cpu_limit(limit=3600*2, recovery_time=120):
    """
    Set limit on the amount of CPU time available to the process.

    Raises resourcelimits.CPULimit when the cpu limit is exceeded, after
    which the program has the number of seconds of recovery time left to
    clean up.
    """
    setrlimit(RLIMIT_CPU, (limit,limit+recovery_time))
    signal(SIGXCPU, _xcpuhandler)

def disk_limit(limit=1e9):
    """
    Sets a maximum file size that can be written for the program.
    """
    setrlimit(RLIMIT_FSIZE, (int(limit), int(limit+2e6)))
    # error caught by normal IO handlers

def memory_limit(limit=2e9):
    """
    Sets a maximum memory limit for the program.
    """
    setrlimit(RLIMIT_DATA, (int(limit), int(limit+2e6)))
