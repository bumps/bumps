import os, sys
import logging
import traceback
import time
import thread
from multiprocessing import Process

from jobqueue import runjob, store
from jobqueue.client import connect

store.ROOT = '/tmp/worker/%s'
DEFAULT_DISPATCHER = 'http://reflectometry.org/queue'
POLLRATE = 10

def log_errors(f):
    def wrapped(*args, **kw):
        try:
            return f(*args, **kw)
        except:
            exc_type,exc_value,exc_trace = sys.exc_info()
            trace = traceback.format_tb(exc_trace)
            message = traceback.format_exception_only(exc_type,exc_value)
            logging.error(message+trace)
    return wrapped

def wait_for_result(remote, id, process, queue):
    """
    Wait for job processing to finish.  Meanwhile, prefetch the next
    request.
    """
    next_request = { 'request': None }
    canceling = False
    while True:
        # Check if process is complete
        process.join(POLLRATE)
        if not process.is_alive(): break

        # Check that the job is still active, and that it hasn't been
        # canceled, or results reported back from a second worker.
        # If remote server is down, assume the job is still active.
        try: response = remote.status(id)
        except: response = None
        if response and response['status'] != 'ACTIVE':
            #print "canceling process"
            process.terminate()
            canceling = True
            break

        # Prefetch the next job; this strategy works well if there is
        # only one worker.  If there are many, we may want to leave it
        # for another worker to process.
        if not next_request['request']:
            # Ignore remote server down errors
            try: next_request = remote.nextjob(queue=queue)
            except: pass

    # Grab results from the store
    try:
        results = runjob.results(id)
    except KeyError:
        if canceling:
            results = { 'status': 'CANCEL', 'message': 'Job canceled' }
        else:
            results = { 'status': 'ERROR', 'message': 'Results not found' }

    #print "returning results",results
    return results, next_request

@log_errors
def update_remote(dispatcher, id, queue, results):
    """
    Update remote server with results.
    """
    #print "updating remote"
    path= store.path(id)
    # Remove results key, if it is there
    try: store.delete(id, 'results')
    except KeyError: pass
    files = [os.path.join(path,f) for f in os.listdir(path)]
    #print "sending results",results
    # This is done with a separate connection to the server so that it can
    # run inside a thread.  That way the server can start the next job
    # while the megabytes of results are being transfered in the background.
    private_remote = connect(dispatcher)
    private_remote.postjob(id=id, results=results, queue=queue, files=files)
    # Clean up files
    for f in files: os.unlink(f)
    os.rmdir(path)

def serve(dispatcher, queue):
    """
    Run the work server.
    """
    assert queue is not None
    next_request = { 'request': None }
    remote = connect(dispatcher)
    while True:
        if not next_request['request']:
            try: next_request = remote.nextjob(queue=queue)
            except: logging.error(traceback.format_exc())
        if next_request['request']:
            jobid = next_request['id']
            if jobid is None:
                logging.error('request has no job id')
                next_request = {'request': None}
                continue
            logging.info('processing job %s'%jobid)
            process = Process(target=runjob.run,
                              args=(jobid,next_request['request']))
            process.start()
            results, next_request = wait_for_result(remote, jobid, process, queue)
            thread.start_new_thread(update_remote,
                                    (dispatcher, jobid, queue, results))
        else:
            time.sleep(POLLRATE)

def main():
    try: os.nice(19)
    except: pass
    if len(sys.argv) <= 1:
        print "Requires queue name"
    queue = sys.argv[1]
    dispatcher = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DISPATCHER
    serve(queue=queue, dispatcher=dispatcher)

if __name__ == "__main__":
    main()
