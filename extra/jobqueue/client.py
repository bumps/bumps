import time
import json
from . import rest

json_content = 'application/json'

class Connection(object):
    def __init__(self, url):
        self.rest = rest.Connection(url)

    def jobs(self, status=None):
        """
        List jobs on the server according to status.
        """
        if status is None:
            response = self.rest.get('/jobs.json')
        else:
            response = self.rest.get('/jobs/%s.json'%status.lower())
        return _process_response(response)['jobs']

    def submit(self, job):
        """
        Submit a job to the server.
        """
        body = json.dumps(job)
        response = self.rest.post('/jobs.json',
                                  mimetype=json_content,
                                  body=body)
        return _process_response(response)

    def info(self, id):
        """
        Return the job structure associated with id.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.get('/jobs/%s.json'%id)
        return _process_response(response)

    def status(self, id):
        """
        Return the job structure associated with id.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.get('/jobs/%s/status.json'%id)
        return _process_response(response)

    def output(self, id):
        """
        Return the result from processing the job.

        Raises ValueError if job not found.
        Raises IOError if communication error.

        Check response['status'] for 'COMPLETE','CANCEL','ERROR', etc.
        """
        response = self.rest.get('/jobs/%s/results.json'%id)
        return _process_response(response)

    def wait(self, id, pollrate=300, timeout=60*60*24):
        """
        Wait for job to complete, returning output.

        *pollrate* is the number of seconds to sleep between checks
        *timeout* is the maximum number of seconds to wait

        Raises IOError if the timeout is exceeded.
        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        start = time.clock()
        while True:
            results = self.output(id)
            #print "waiting: result is",results
            if results['status'] in ('PENDING', 'ACTIVE'):
                #print "waiting for job %s"%id
                if time.clock() - start > timeout:
                    raise IOError('job %s is still pending'%id)
                time.sleep(pollrate)
            else:
                #print "status for %s is"%id,results['status'],'- wait complete'
                return results

    def stop(self, id):
        """
        Stop the job.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.post('/jobs/%s?action=stop'%id)
        return _process_response(response)

    def delete(self, id):
        """
        Delete the job and all associated files.

        Raises ValueError if job not found.
        Raises IOError if communication error.
        """
        response = self.rest.delete('/jobs/%s.json'%id)
        return _process_response(response)

    def nextjob(self, queue):
        """
        Fetch the next job to process from the queue.
        """
        # TODO: combine status check and prefetch to reduce traffic
        # TODO: worker sends active and pending jobs so we can load balance
        body = json.dumps({'queue': queue})
        response = self.rest.post('/jobs/nextjob.json',
                                  mimetype=json_content,
                                  body=body)
        return _process_response(response)

    def postjob(self, queue, id, results, files):
        """
        Return results from a processed job.
        """
        # TODO: sign request
        fields = {'queue': queue, 'results': json.dumps(results)}
        response = self.rest.postfiles('/jobs/%s/postjob'%id,
                                       files=files,
                                       fields=fields)
        return _process_response(response)

    def putfiles(self, id, files):
        # TODO: sign request
        response = self.rest.putfiles('/jobs/%s/data/'%id,
                                      files=files)
        return _process_response(response)

def _process_response(response):
    headers, body = response
    #print "response",response[body]
    if headers['status'] == '200':
        return json.loads(body)
    else:
        err = headers['status']
        msg = rest.RESPONSE.get(err,("Unknown","Unknown code"))[1]
        raise IOError("server response %s %s"%(err,msg))

def connect(url):
    return Connection(url)
