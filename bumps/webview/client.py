"""
Remote bumps api access.

Usage::

    import bumps.names as bp

    client = bp.remote_bumps(url=...)
    client.set_fit_problem(problem)
    ...
    client.disconnect()
"""

import asyncio
import inspect
import json
import socketio

from bumps.api import REGISTRY

# ── configuration ─────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8502
N_ITERATIONS = 15
FIT_SECONDS = 30  # wall time to let each fit run before stopping
POLL_INTERVAL = 2.0  # seconds between active_fit polls


class BumpsClient:
    """
    Proxy for the bumps server api with methods for each api call.

    *url* is the url of the running bumps server.
    """

    def __init__(self, url: str):
        self.url = url
        self._sio = socketio.AsyncClient(reconnection=True)
        self.verbose = False

    async def connect(self) -> None:
        """
        Connect to the client.
        """
        await self._sio.connect(self.url)
        print(f"Connected to {self.url}")

    async def disconnect(self) -> None:
        """Disconnect from client."""
        await self._sio.disconnect()

    async def __aenter__(self):
        await self._client.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._client.disconnect()

    async def _call(self, method, *args, silent=False):
        import msgpack

        data = tuple(args) if args else None
        result = await self._sio.call(method, data, timeout=120)
        if isinstance(result, (bytes, bytearray)):
            try:
                result = msgpack.unpackb(result, raw=False)
            except Exception:
                try:
                    result = json.loads(result.decode())
                except Exception:
                    pass
        if self.verbose:
            print(f"    [{method}] -> {result!r}")
        return result

    async def wait_for_fit(self, timeout: float = FIT_SECONDS):
        """Poll active_fit until empty (fit complete) or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            active = await self.get_shared_setting("active_fit")
            if not active:
                print("  fit complete.")
                return True
            step = active.get("step", "?") if isinstance(active, dict) else "?"
            chisq = active.get("chisq", "?") if isinstance(active, dict) else "?"
            print(f"  fitting... step={step} chisq={chisq}")
            if asyncio.get_event_loop().time() > deadline:
                print("  timeout — stopping fit.")
                await self.stop_fit(True)  # wait=True: blocks until thread exits
                return False
            await asyncio.sleep(POLL_INTERVAL)


def _populate_client_api() -> None:
    """
    Walk ``bumps.api.REGISTRY`` and create a thin async method on *cls*
    for each entry.  The created method:

    * has the exact signature (including default values) of the server‑side function,
    * carries the original doc‑string,
    * forwards the call to ``self.call(name, *args, **kwargs)`` and returns the result.
    * is marked as ``async`` so it can be awaited just like the original RPC.
    """

    # The “self” argument – a normal positional‑or‑keyword parameter.
    self_param = inspect.Parameter(
        name="self",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,  # regular positional argument
        default=inspect.Parameter.empty,  # no default value
        annotation=inspect.Parameter.empty,  # we don’t need a type hint for self
    )

    def _add_self(sig: inspect.Signature) -> inspect.Signature:
        """Adds *self* to the start of the method signature."""
        new_params = [self_param, *list(sig.parameters.values())]
        # Preserve the original return annotation
        return sig.replace(parameters=new_params)

    for name, fn in REGISTRY.items():
        # Build a dummy function that forwards to ``self._client.call``.
        # We’ll later replace its ``__code__`` with a generated one that
        # respects the signature.
        async def _forward(self, *args, _method=name, **kwargs):
            return await self._call(_method, *args, **kwargs)

        # Give the forwarder the right signature and docstring.
        # Include an additional "self" argument since the proxy lives as
        # a method of BumpsClient rather than a function in the api module.
        try:
            new_sig = _add_self(inspect.signature(fn))
        except Exception:
            # If the function is a built‑in or otherwise non‑inspectable,
            # fall back to a generic *args/**kwargs signature.
            new_sig = inspect.signature(lambda self, *a, **kw: None)
        doc = inspect.getdoc(fn) or ""

        # Attach our synthetic signature.
        _forward.__signature__ = new_sig  # type: ignore[attr-defined]
        _forward.__doc__ = doc

        # Bind the method into the instance namespace.
        setattr(BumpsClient, name, _forward)


_populate_client_api()


async def remote_bumps(url: str | None = None) -> BumpsClient:
    """
    Open a client connection to a remote bumps server.

    If *url* is not provided then start a new bumps server.
    """
    if url is None:
        from .webserver import start_bumps, get_server_url

        await start_bumps()
        url = get_server_url()

    client = BumpsClient(url)
    await client.connect()
    return client
