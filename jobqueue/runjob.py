import os
import sys
import traceback

from . import store
from . import services

try:
    from . import resourcelimits
    def setlimits():
        resourcelimits.cpu_limit(2*3600,120)
        resourcelimits.disk_limit(1e9)
        resourcelimits.memory_limit(2e9)
except:
    def setlimits(): pass


def build_command(id, request):
    """
    Build a script file to run the service and return the command
    needed to start it.

    The script will be in the job execution path.

    The command includes the python interpreter name of the caller.

    The resulting command can be used from "srun" within the slurm queue.
    """
    path = store.path(id)
    script = """
import os
from jobqueue import runjob, store
#import sys; print "\\n".join(sys.path)
id = "%s"
request = store.get(id,"request")
runjob.run(id, request)
"""%id
    scriptfile = os.path.join(path,'runner.py'%id)
    open(scriptfile,'w').write(script)
    return sys.executable+" "+scriptfile

def run(id, request):
    """
    Load a service and run the request.


    """
    try:
        result = {
              'status': 'COMPLETE',
              'result': _run(id, request),
              }
    except:
        # Trim the traceback to exclude run and _run.
        exc_type,exc_value,exc_trace = sys.exc_info()
        relevant_list = traceback.extract_tb(exc_trace)[2:]
        message = traceback.format_exception_only(exc_type,exc_value)
        trace = traceback.format_list(relevant_list)
        result = {
              'status': 'ERROR',
              'error': "".join(message).rstrip(),
              'trace': "".join(trace).rstrip(),
            }
    store.put(id,'results',result)

def _run(id, request):
    # Prepare environment
    #print "\n".join(sys.path)
    path = store.path(id)
    store.create(id)  # Path should already exist, but just in case...
    os.chdir(path)    # Make sure the program starts in the path
    sys.stdout = open(os.path.join(path,'stdout.txt'),'w')
    sys.stderr = open(os.path.join(path,'stderr.txt'),'w')
    setlimits()

    # Run service
    service = getattr(services, request['service'], None)
    if service is None:
        raise ValueError("service <%s> not available"%request['service'])
    else:
        return service(request)

def results(id):
    results = store.get(id,'results')
    if results is None:
        raise RuntimeError("Results for %d cannot be empty"%id)
    return results
