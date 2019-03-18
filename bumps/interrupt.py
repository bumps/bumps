try:
    from signal import signal, SIGTERM, SIGINT
except ImportError:
    # For systems that don't use unix signals, just pretend
    SIGTERM = SIGINT = -1
    def signal(signum, handler):
        pass

class _Interrupt(object):
    """
    Turn SIGTERM into a KeyboardInterrupt error on systems which support
    signal handling.
    """
    def __init__(self):
        self._level = 0
        self.interrupted = False
        signal(SIGINT, self._signal_handler)
        signal(SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        # On interrupt, remember that an interrupt has been signalled.
        self.interrupted = True
        if self._level > 0:
            raise KeyboardInterrupt("Received signal %s"%signum)

    def __enter__(self):
        # Signal that a keyboard interrupt should be generated when the
        # exception happens, in addition to setting the interrupted flag.
        self._level += 1

    def __exit__(self, exception_type, exception_value, traceback):
        self._level -= 1
        # Eat the exception if it is a keyboard interrupt, otherwise
        # let it through.
        return exception_type == KeyboardInterrupt

_INTERRUPT = None
def set_traps():
    global _INTERRUPT
    if _INTERRUPT is None:
        _INTERRUPT = _Interrupt()

def interruptable():
    """
    Context manager which exits on SIGINT or SIGTERM.

    Call interrupt.set_trap() to prepare for interrupts.

    Call interrupt.interrupted() to test whether the interrupt was triggered
    in a block of code.
    """
    if _INTERRUPT is None:
        raise RuntimeError("Call interrupt.set_traps() first")
    return _INTERRUPT

def interrupted():
    if _INTERRUPT is None:
        raise RuntimeError("Call interrupt.set_traps() first")
    return _INTERRUPT.interrupted

def demo():
    import time
    set_traps()
    for k in range(10):
        with interruptable():
            print("do stuff", k)
            time.sleep(5)
        if interrupted():
            print("stuff was interrupted")
            break
        print("record completed work")
        time.sleep(5)
        if interrupted():
            print("interrupt suppressed during recording")
            break

if __name__ == "__main__":
    demo()
