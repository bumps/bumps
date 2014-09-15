# This program is in the public domain
"""
Progress monitors.

Process monitors accept a :class:`bumps.history.History` object each cycle
and perform some sort of work.
"""
from __future__ import print_function

__all__ = ['Monitor', 'Logger', 'TimedUpdate']

from numpy import inf


class Monitor(object):
    """
    Base class for monitors.
    """
    def config_history(self, history):
        """
        Indicate which fields are needed by the monitor and for what duration.
        """
        pass

    def __call__(self, history):
        """
        Give the monitor a new piece of history to work with.
        """
        pass


def _getfield(history, field):
    """
    Return the last value in the trace, or None if there is no
    last value or no trace.
    """
    trace = getattr(history, field, [])
    try:
        return trace[0]
    except IndexError:
        return None


class Logger(Monitor):
    """
    Keeps a record of all values for the desired fields.

    *fields* is a list of history fields to store.

    *table* is an object with a *store(field=value,...)* method, which gets
    the current value of each field every time the history is updated.

    Call :meth:`config_history` with the :class:`bumps.history.History`
    object before starting so that the correct fields are stored.
    """
    def __init__(self, fields=(), table=None):
        self.fields = fields
        self.table = table

    def config_history(self, history):
        """
        Make sure history records each logged field.
        """
        kwargs = dict((key, 1) for key in self.fields)
        history.requires(**kwargs)

    def __call__(self, history):
        """
        Record the next piece of history.
        """
        record = dict((f, _getfield(history, f)) for f in self.fields)
        self.table.store(**record)


class TimedUpdate(Monitor):
    """
    Indicate progress every n seconds.

    The process should provide time, value, point, and step to the
    history update. Call :meth:`config_history` with the
    :class:`bumps.history.History` object before starting so that
    these fields are stored.

    *progress* is the number of seconds to go before showing progress, such
    as time or step number.

    *improvement* is the number of seconds to go before showing
    improvements to value.

    By default, the updater only prints step number and improved value.
    Subclass TimedUpdate with replaced :meth:`show_progress` and
    :meth:`show_improvement` to trigger GUI updates or show parameter
    values.
    """
    def __init__(self, progress=60, improvement=5):
        self.progress_delta = progress
        self.improvement_delta = improvement
        self.progress_time = -inf
        self.improvement_time = -inf
        self.value = inf
        self.improved = False

    def config_history(self, history):
        history.requires(time=1, value=1, point=1, step=1)

    def show_improvement(self, history):
        print("step", history.step, "value", history.value)

    def show_progress(self, history):
        pass

    def __call__(self, history):
        t = history.time[0]
        v = history.value[0]
        if v < self.value:
            self.improved = True
        if t > self.progress_time + self.progress_delta:
            self.progress_time = t
            self.show_progress(history)
        if self.improved and t > self.improvement_time + self.improvement_delta:
            self.improved = False
            self.improvement_time = t
            self.show_improvement(history)
