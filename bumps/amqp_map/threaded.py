# This program is public domain
# Author: Paul Kienzle
"""
Thread and daemon decorators.

See :function:`threaded` and :function:`daemon` for details.
"""

from functools import wraps
import itertools
import threading

#TODO: fix race conditions
# notify may be called twice in after()
# 1. main program calls fn() which starts the processing and returns job
# 2. main program calls job.after(notify)
# 3. after() suspends when __after is set but before __stopped is checked
# 4. thread ends, setting __stopped and calling __after(result)
# 5. main resumes, calling __after(result) since __stoped is now set
# solution is to use thread locks when testing/setting __after.
_after_lock = threading.Lock()
class AfterThread(threading.Thread):
    """
    Thread class with additional 'after' capability which runs a function
    after the thread is complete.  This allows us to separate the notification
    from the computation.

    Unlike Thread.join, the wait() method returns the value of the computation.
    """
    name = property(threading.Thread.getName,
                    threading.Thread.setName,
                    doc="Thread name")
    def __init__(self, *args, **kwargs):
        self.__result = None
        self.__after = kwargs.pop('after',None)
        threading.Thread.__init__(self, *args, **kwargs)

    def after(self, notify=None):
        """
        Calls notify after the thread is complete.  Notify should
        take a single argument which is the result of the function.

        Note that notify will be called from the main thread if the
        thread is already complete when thread.after(notify) is called,
        otherwise it will be called from thread.
        """
        _after_lock.acquire()
        self.__after = notify
        # Run immediately if thread is already complete
        if self._Thread__started and self._Thread__stopped:
            post = notify
        else:
            post = lambda x: x
        _after_lock.release()
        post(self.__result)

    def run(self):
        """
        Run the thread followed by the after function if any.
        """
        if self._Thread__target:
            self.__result = self._Thread__target(*self._Thread__args,
                                                 **self._Thread__kwargs)
            _after_lock.acquire()
            if self.__after is not None:
                post = self.__after
            else:
                post = lambda x: x
            _after_lock.release()
            post(self.__result)

    def wait(self, timeout=None):
        """
        Wait for the thread to complete.

        Returns the result of the computation.

        Example::

            result = thread.wait()

        If timeout is used, then wait() may return before the result is
        available.  In this case, wait() will return None.  This can be
        used as follows::

            while True:
                result = thread.wait(timeout=0)
                if result is not None: break
                ... do something else while waiting ...

        Timeout should not be used with functions that may return None.
        This is due to the race condition in which the thread completes
        between the timeout triggering in wait() and the main thread
        calling thread.isAlive().
        """
        self.join(timeout)
        return self.__result

def threaded(fn):
    """
    @threaded decorator for functions to be run in a thread.

    Returns the running thread.

    The returned thread supports the following methods::

        wait(timeout=False)
            Waits for the function to complete.
            Returns the result of the function if the thread is joined,
            or None if timeout.  Use thread.isAlive() to test for timeout.
        after(notify)
            Calls notify after the thread is complete.  Notify should
            take a single argument which is the result of the function.
        isAlive()
            Returns True if thread is still running.
        name
            Thread name property.  By default the name is 'fn-#' where fn
            is the function name and # is the number of times the thread
            has been invoked.

    For example::

        @threaded
        def compute(self,input):
            ...
        def onComputeButton(self,evt):
            thread = self.compute(self.input.GetValue())
            thread.after(lambda result: wx.Post(self.win,wx.EVT_PAINT))

    A threaded function can also be invoked directly in the current thread::

        result = self.compute.main(self.input.GetValue())

    All threads must complete before the program can exit.  For queue
    processing threads which wait are alive continuously waiting for
    new input, use the @daemon decorator instead.
    """
    instance = itertools.count(1)
    @wraps(fn)
    def wrapper(*args, **kw):
        name = "%s-%d"%(fn.func_name,instance.next())
        thread = AfterThread(target=fn,args=args,kwargs=kw,name=name)
        thread.start()
        return thread
    wrapper.main = fn
    return wrapper

def daemon(fn):
    """
    @daemon decorator for functions to be run in a thread.

    Returns the running thread.

    Unlike threaded functions, daemon functions are not expected to complete.
    """
    instance_counter = itertools.count(1)
    @wraps(fn)
    def wrapper(*args, **kw):
        name = "%s-%d"%(fn.func_name,instance_counter.next())
        thread = threading.Thread(target=fn,args=args,kwargs=kw,name=name)
        thread.setDaemon(True)
        thread.start()
        return thread
    wrapper.main = fn
    return wrapper
