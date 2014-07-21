# This program is in the public domain
# Author: Paul Kienzle
"""
Log of progress through a computation.

Each cycle through a computation, a process can update its history,
adding information about the number of function evaluations, the
total time taken, the set of points evaluated and their values, the
current best value and so on.  The process can use this history
when computing the next set of points to evaluate and when checking
if the termination conditions are met.  Any values that may be
useful outside the computation, e.g., for logging or for updating
the user, should be recorded.  In the ideal case, the history
is all that is needed to restart the process in case of a system
crash.

History consists of a set of traces.  The content of the traces
themselves is provided by the computation, but various stake holders
can use them.  For example, the user may wish to log the set of points
that have been evaluated and their values using the system logger
and an optimizer may require a certain amount of history to calculate
the next set of values.

New traces are provided using :meth:`History.provides`.  For example,
the following adds traces for 'value' and 'point' to the history, and
requires the best value on the two previous cycles in order to do its work:

    >>> from bumps.history import History
    >>> h = History()
    >>> h.provides(value=2, point=0)

Initially the history is empty:

    >>> print(len(h.value))
    0

After three updates we see that only two values are kept:

    >>> h.update(value=2,point=[1,1,1])
    >>> h.update(value=1,point=[1,0.5,1])
    >>> h.update(value=0.5,point=[1,0.5,0.9])
    >>> print(h.value)
    Trace value: 0.5, 1
    >>> print(len(h.value))
    2

Note that point is not monitored since it is not required:

    >>> print(h.point[0])
    Traceback (most recent call last):
        ...
    IndexError: point has not accumulated enough history

Traces may be used as accumulators.  In that case, the next
value is added to the tail value before appending to the trace.
For example:

    >>> h = History()
    >>> h.provides(step=1)
    >>> h.accumulate(step=1)
    >>> h.accumulate(step=1)
    >>> print(h.step[0])
    2
"""

# Design questions:
# 1. Can optimizer function evaluators add traces?  Can they use traces?
# 2. Do we want to support a skip option on traces, so that only every nth
#    item is preserved?  This is probably too hard.


class History(object):

    """
    Collection of traces.

    Provided traces can be specified as key word arguments, name=length.
    """

    def __init__(self, **kw):
        self.provides(**kw)

    def provides(self, **kw):
        """
        Specify additional provided fields.

        Raises AttributeError if trace is already provided or if the trace
        name matches the name of one of the history methods.
        """
        for k, v in kw.items():
            # Make sure the additional trait is not already provided.
            # This test should also catch methods such as provides/requires
            # and static properties such as bounds that are set from outside.
            if hasattr(self, k):
                raise AttributeError("history already provides " + k)
            else:
                mon = self._new_trace(keep=v, name=k)
                setattr(self, k, mon)

    def requires(self, **kw):
        """
        Specify required fields, and their history length.
        """
        for k, v in kw.items():
            try:
                mon = getattr(self, k)
                mon.requires(v)
            except AttributeError:
                raise AttributeError("history does not provide " + k
                                     + "\nuse one of " + self._trace_names())

    def accumulate(self, **kw):
        """
        Extend the given traces with the provided values.  The traced
        value will be the old value plus the new value.
        """
        for k, v in kw.items():
            try:
                getattr(self, k).accumulate(v)
            except AttributeError:
                raise AttributeError(k + " is not being traced")

    def update(self, **kw):
        """
        Extend the given traces with the provided values.  The traced
        values are independent.  Use accumulate if you want to add the
        new value to the previous value in the trace.
        """
        for k, v in kw.items():
            try:
                getattr(self, k).put(v)
            except AttributeError:
                raise AttributeError(k + " is not being traced")

    def clear(self):
        """
        Clear history, removing all traces
        """
        self.__dict__.clear()

    def _new_trace(self, keep=None, name=None):
        """
        Create a new trace.  We use a factory method here so that
        the History subclass can control the kind of trace created.
        The returned trace must be a subclass of history.Trace.
        """
        return Trace(keep=keep, name=name)

    def _traces(self):
        return [trace
                for trace in self.__dict__.values()
                if isinstance(trace, Trace)]

    def _trace_names(self):
        traces = [trace.name for trace in self._traces()]
        return ", ".join(l for l in sorted(traces))

    def __str__(self):
        traces = sorted(self._traces(), lambda x, y: cmp(x.name, y.name))
        return "\n".join(str(l) for l in traces)

    def snapshot(self):
        """
        Return a dictionary of traces { 'name':  [v[n], v[n-1], ..., v[0]] }
        """
        return dict((trace.name, trace.snapshot()) for trace in self._traces())

    def restore(self, state):
        """
        Restore history to the state returned by a call to snapshot
        """
        for k, v in state.items():
            try:
                getattr(self, k).restore(v)
            except KeyError:
                pass


class Trace(object):

    """
    Value trace.

    This is a stack-like object with items inserted at the beginning, and
    removed from the end once the maximum length *keep* is reached.

    len(trace) returns the number of items in the trace
    trace[i] returns the ith previous element in the history
    trace.requires(n) says how much history to keep
    trace.put(value) stores value
    trace.accumulate(value) adds value to the previous value before storing
    state = trace.snapeshot() returns the values as a stack, most recent last
    trace.restore(state) restores a snapshot

    Note that snapshot/restore uses lists to represent numpy arrays, which
    may cause problems if the trace is capturing lists.
    """
    # Implementation note:
    # Traces are stored in reverse order because append is faster than insert.
    # This detail is hidden from the caller since __getitem__ returns the
    # appropriate value.
    # TODO: convert to circular buffer unless keeping the full trace
    # TODO: use numpy arrays for history

    def __init__(self, keep=1, name="trace"):
        self.keep = keep
        self._storage = []
        self.name = name

    def requires(self, n):
        """
        Set the trace length to be at least n.
        """
        # Note: never shorten the trace since another algorithm, condition,
        # or monitor may require the longer trace.
        if n > self.keep:
            self.keep = n

    def accumulate(self, value):
        if self.keep < 1:
            return
        try:
            value = self._storage[-1] + value
        except IndexError:
            pass  # Value is 0 + value => 0
        self.put(value)

    def put(self, value):
        """
        Add an item to the trace, shifting off from the beginning
        when the trace is full.
        """
        if self.keep < 1:
            return
        if len(self._storage) == self.keep:
            self._storage = self._storage[1:]
        self._storage.append(value)

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        if key < 0:
            raise IndexError(self.name
                             + " can only be accessed from the beginning")
        try:
            return self._storage[-key - 1]
        except IndexError:
            raise IndexError(self.name + " has not accumulated enough history")

    def __setitem__(self, key, value):
        raise TypeError("cannot write directly to a trace; use put instead")

    def __str__(self):
        return ("Trace " + self.name + ": "
                + ", ".join([str(k) for k in reversed(self._storage)]))

    def snapshot(self):
        """
        Capture state of the trace.

        Numpy arrays are converted to lists so that the trace can be easily
        converted to json.
        """
        import numpy as np
        if isinstance(self._storage[0], np.ndarray):
            return [v.tolist() for v in self._storage]
        else:
            return self._storage[:]

    def restore(self, state):
        """
        Restore a trace from a captured snapshot.

        Lists are converted to numpy arrays.
        """
        import numpy as np
        if isinstance(state[0], list):
            state = [np.asarray(v) for v in state]
        if len(state) > self.keep:
            self._storage = state[-self.keep:]
        else:
            self._storage = state[:]
