import os
import time

from jobqueue import dispatcher, db, store, notify

db.DEBUG = False
DEBUG = False

path = os.path.abspath(os.path.dirname(__file__))
URI = "sqlite:///%s/test.db"%path

def setupdb(uri):
    db.DB_URI = uri
    store.ROOT = "/tmp/test_store/%s"
    if uri.startswith("sqlite"):
        try: os.unlink(uri[10:])
        except:
            print "could not unlink",uri[10:]
            pass
    queue = dispatcher.Scheduler()
    return queue

def checkspeed(uri=URI):
    # Isolate the cost of database access
    store.create = lambda *args: {}
    store.get = lambda *args: {}
    store.put = lambda *args: {}
    notify.notify = lambda *args, **kw: None
    queue = setupdb(uri)
    testj = { 'name' : 'test1', 'notify' : 'me' }
    t = time.time()
    for i in range(80):
        for j in range(10):
            testj['notify'] = 'me%d'%j
            queue.submit(testj, origin="here%d"%j)
        for j in range(10):
            request = queue.nextjob(queue='cue')
            queue.postjob(1, {'status': 'COMPLETE', 'result': 0})
        print 10*(i+1), time.time()-t
        t = time.time()

def test(uri=URI):

    queue = setupdb(uri)
    def checkqueue(pending=[], active=[], complete=[]):
        qpending = queue.jobs('PENDING')
        qactive = queue.jobs('ACTIVE')
        qcomplete = queue.jobs('COMPLETE')
        if DEBUG: print "pending",qpending,"active",qactive,"complete",qcomplete
        assert pending == qpending
        assert active == qactive
        assert complete == qcomplete


    test1 = { 'name' : 'test1', 'notify' : 'me' }
    test2 = { 'name' : 'test2', 'notify' : 'me' }
    test3 = { 'name' : 'test3', 'notify' : 'you' }

    # No jobs available for running initially
    checkqueue([],[],[])
    request = queue.nextjob(queue='cue')
    if DEBUG: print "nextjob",request
    assert request['request'] is None

    #jobs = queue.jobs()
    #print "initially empty job list", jobs
    #assert jobs == []
    job1 = queue.submit(test1, origin="here")
    if DEBUG: print "first job id",job1
    assert job1 == 1
    job2 = queue.submit(test2, origin="here")
    assert job2 == 2
    checkqueue([1,2],[],[])

    if DEBUG: print "status(0)",queue.status(1)
    assert queue.status(1) == 'PENDING'

    if DEBUG: print "status(3)",queue.status(3)
    assert queue.status(3) == 'UNKNOWN'

    if DEBUG: print "info(0)", queue.info(1)
    assert queue.info(1)['name'] == test1['name']

    request = queue.nextjob(queue='cue')
    if DEBUG: print "nextjob",request
    assert request['request']['name'] == test1['name']
    checkqueue([2],[1],[])

    job2 = queue.submit(test3, origin="there")
    request = queue.nextjob(queue='cue')
    if DEBUG: print "nextjob",request
    assert request['request']['name'] == test3['name']
    checkqueue([2],[1,3],[])

    queue.postjob(1, {'status': 'COMPLETE', 'result': 0})
    checkqueue([2],[3],[1])

if __name__ == "__main__":
    test()
    #checkspeed()
