import threading
from multiprocessing import Process

from . import runjob, jobid, store

class Scheduler(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._nextjob = threading.Event()
        self._jobs = []
        self._pending = []
        self._info = {}
        self._status = {}
        self._results = {}
        self._jobmonitor = threading.Thread(target=self._run_queue)
        self._jobmonitor.start()
        self._current_id = None
    def _run_queue(self):
        while True:
            self._nextjob.wait()
            with self._lock:
                if not self._pending:
                    self._nextjob.clear()
                    continue
                self._current_id = self._pending.pop(0)
                self._status[self._current_id] = 'ACTIVE'
                request = self._info[self._current_id]
                self._stopping = None
                self._current_process = Process(target=runjob.run,
                                                args=(self._current_id,request))
            self._current_process.start()
            self._current_process.join()
            results = runjob.results(self._current_id)
            with self._lock:
                self._results[self._current_id] = results
                self._status[self._current_id] = results['status']

    def jobs(self, status=None):
        with self._lock:
            if status is None:
                response = self._jobs[:]
            else:
                response = [j for j in self._jobs if self._status[j] == status]
        return response
    def submit(self, request, origin):
        with self._lock:
            id = int(jobid.get_jobid())
            store.create(id)
            store.put(id,'request',request)
            request['id'] = id
            self._jobs.append(id)
            self._info[id] = request
            self._status[id] = 'PENDING'
            self._results[id] = {'status':'PENDING'}
            self._pending.append(id)
            self._nextjob.set()
        return id
    def results(self, id):
        with self._lock:
            return self._results.get(id,{'status':'UNKNOWN'})
    def status(self, id):
        with self._lock:
            return self._status.get(id,'UNKNOWN')
    def info(self, id):
        with self._lock:
            return self._info[id]
    def cancel(self, id):
        with self._lock:
            try: self._pending.remove(id)
            except ValueError: pass
            if self._current_id == id and not self._stopping == id:
                self._stopping = id
                self._current_process.terminate()
            self._status[id] = 'CANCEL'
    def delete(self, id):
        self.cancel(id)
        with self._lock:
            try: self._jobs.remove(id)
            except ValueError: pass
            self._info.pop(id, None)
            self._results.pop(id, None)
            self._status.pop(id, None)
        store.destroy(id)
