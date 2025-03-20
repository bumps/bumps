# play with function closure
from dataclasses import dataclass
from typing import Any, Dict
from itertools import chain
import inspect
from copy import deepcopy
import builtins

from numpy import ndarray
import numpy as np
import dill

_safe_names = [
    "__build_class__",
    "None",
    "False",
    "True",
    "abs",
    "bool",
    "bytes",
    "callable",
    "chr",
    "complex",
    "divmod",
    "float",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "oct",
    "ord",
    "pow",
    "range",
    "repr",
    "round",
    "slice",
    "sorted",
    "str",
    "tuple",
    "zip",
]

_safe_exceptions = [
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BufferError",
    "BytesWarning",
    "DeprecationWarning",
    "EOFError",
    "EnvironmentError",
    "Exception",
    "FloatingPointError",
    "FutureWarning",
    "GeneratorExit",
    "IOError",
    "ImportError",
    "ImportWarning",
    "IndentationError",
    "IndexError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "NameError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "ReferenceError",
    "RuntimeError",
    "RuntimeWarning",
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
    "SystemError",
    "SystemExit",
    "TabError",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
]

BASE_CONTEXT = {name: getattr(builtins, name) for name in _safe_names + _safe_exceptions}
SAFE_TYPES = (ndarray, float, int, str, complex)


def safe(obj):
    # print(f"checking {obj} in {SAFE_TYPES}")
    if isinstance(obj, SAFE_TYPES):
        return True
    if isinstance(obj, (tuple, list)):
        return all(safe(v) for v in obj)
    if isinstance(obj, dict):
        return all(safe(k) and safe(v) for k, v in obj.items())
    # print(f"{obj} is not safe")
    return False


@dataclass
class UserFunction:
    name: str
    source: str
    context: Dict[str, Any]

    def __init__(self, name, source, context):
        # print("restoring", name)
        self.name = name
        self.source = source
        self.context = context
        self._fn = None

    @property
    def fn(self):
        # TODO: maybe use RestrictedPython to compile the source
        # TODO: warn user that model contains untrusted source
        if self._fn is None:
            if self.source:
                # TODO: hostile functions can do arbitrary things with attribute refs
                global_context = {
                    **BASE_CONTEXT,
                    **self.context,
                }
                # print("compiling")
                # print(self.source)
                # print(global_context)
                local_context = {}
                # print(f"compiling <{self.source}> in", self.context)
                exec(self.source, global_context, local_context)
                # print("local context", local_context)
                self._fn = local_context[self.name]
            else:  # Matches None and ""
                self._fn = dill.loads(self.context["pickle"])
        return self._fn

    def __call__(self, *args, **kw):
        return self.fn(*args, **kw)


def check_context(context):
    return not all(safe(v) for _, v in context.items())


def capture_context(fn):
    """
    Raises TypeError if the context is unsafe
    """
    context = {}
    try:
        closure = inspect.getclosurevars(fn)
        # print("closure =>", closure)
        iter_items = chain(closure.globals.items(), closure.nonlocals.items())
    except Exception as exc:
        # TODO: numba support requires source inspection to find closure
        # print(f"*** no closure captured for {fn.__name__} ***", exc)
        iter_items = []
        # raise
    # build context from closure
    # ignoring builtins and unbound for now
    for k, v in iter_items:
        # TODO: maybe recurse over functions? classes?
        if k not in BASE_CONTEXT:
            # print(f"enclosing {k} = {v}")
            context[k] = deepcopy(v)  # TODO: Do we need deepcopy?
    return context


def capture_function(fn):
    if isinstance(fn, UserFunction):
        return fn
    if fn is None:
        return None
    # print("capturing", fn)

    name = fn.__name__
    # print("type fn", type(fn))
    # Note: need dedent to handle decorator syntax. Dedent will fail when there are
    # triple-quoted strings. Alternative: if first character is a space, then wrap
    # the code in an "if True:" block
    # source = dedent(inspect.getsource(fn)) #.strip()
    source = inspect.getsource(fn)
    if source[0] in " \t":
        source = "if True:\n" + source
    # print("source =>", source)

    try:
        context = capture_context(fn)
    except TypeError as exc:
        context = None
        print(f"Encountered {exc}")
        # TODO: fall back to dill

    if context is None:
        # Not trapping errors here because I don't know what to do with them.
        pickle = dill.dumps(fn)
        # Empty source string is the sigil for the dill pickle
        capture = UserFunction(name, "", {"pickle": pickle})
    else:
        capture = UserFunction(name, source, context)
    capture._fn = fn  # already have the function; don't need to recompile
    return capture


from numpy import minimum, arange, sin, empty_like

cutoff = 5


def topf(x, a=1, b=5):
    return minimum(a * x + b, cutoff)


def demo():
    def roundtrip(fn, kwargs):
        import inspect

        value = fn(**kwargs)

        # print("1")
        capture = capture_function(fn)
        print("source => ", capture.source)
        print("context => ", capture.context)
        # assert (value == capture.fn(**kwargs)).all()

        # Force recompile
        capture._fn = None
        assert (value == capture.fn(**kwargs)).all()

    kwargs = dict(a=15, b=3, x=arange(5))

    print("\n== simple ==")

    def f(x, a=1, b=5):
        return a * x + b

    roundtrip(f, kwargs)

    # Can't do lambda because the function object doesn't have a name
    # Because inspect.getsource is so stupid, we would have to extract the
    # lambda expression from a source line such as f = lambda x, a, b: ...
    # or roundtrip(lambda x, a, b: ..., kwargs). Can't do either of these
    # without a full parser.
    if 0:
        print("\n== lambda ==")
        f = lambda x, a=1, b=5: a * x + b
        roundtrip(f, kwargs)
        print("\n== lambda ==")
        roundtrip(
            lambda x, a=1, b=5: a * x + b,
            kwargs,
        )

    print("\n== function ==")

    def f(x, a=1, b=5):
        return abs(sin(a * x + b))

    roundtrip(f, kwargs)

    print("\n== closure over locals ==")
    cutoff = 36

    def f(x, a=1, b=5):
        return minimum(a * x + b, cutoff)

    roundtrip(f, kwargs)
    cutoff = 15
    roundtrip(f, kwargs)

    # TODO: test cutoff with nested dicts, lists, arrays

    print("\n== closure over globals ==")
    roundtrip(topf, kwargs)

    if 0:
        # TODO: add numba support
        print("\n== numba jit with prange ==")
        from numba import njit, prange

        # Numba doesn't work with closures. It uses the value
        # as defined when njit was called. I don't see anything
        # obvious in the jitted object that contains the context.
        # cutoff = 4
        @njit(parallel=True)
        def f(x, a=1, b=5):
            cutoff = 4
            result = empty_like(x)
            for k in prange(len(x)):
                if x[k] < cutoff:
                    result[k] = a * x[k] + b
                else:
                    result[k] = cutoff
            return result

        # print(f(**kwargs))
        # print("jit", dir(f))
        # print(f.py_func.__code__)
        roundtrip(f, kwargs)

    try:
        from scipy.special import gammainc

        def f(x, a, b):
            return gammainc(a, b)

        roundtrip(f, kwargs)
        # print("!!!! should have flagged unsafe function gammainc")
    except Exception as exc:
        print(f"Correctly raised {exc}")

    # Left over from testing AST based source check
    def hello(x, a=1, b=2, c=3):
        global __builtins__
        nonlocal roundtrip
        import pandas as pd
        from numpy import sin

        y = x.__module__.__builtins

    # source = dedent(inspect.getsource(hello))
    # print(inspect.getsource(hello))
    # print(ast.dump(ast.parse(source), indent=2))
    # print(f"warnings:\n  {'\n  '.join(check_source(source))}")

    # pack = capture_function(hello)

    # helper functions
    # class method
    # instance method
    # helper class
    # third party packages like pandas


if __name__ == "__main__":
    demo()
