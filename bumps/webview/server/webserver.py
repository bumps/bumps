import asyncio
import functools
import mimetypes
import os
import socket
from pathlib import Path
from typing import Callable, Optional, Union, List

import matplotlib

# from .main import setup_bumps
from . import cli
from . import api
from . import persistent_settings
from .logger import logger
from .cli import BumpsOptions

matplotlib.use("agg")

mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

TRACE_MEMORY = False
PREFERRED_PORT = 5148  # "SLAB"

# can get by name and not just by id


async def index(request):
    """Serve the client-side application."""
    from aiohttp import web

    client = api.state.client_path
    index_path = client / "dist" / "index.html"
    if not index_path.exists():
        return web.Response(
            body=f"<h2>Client not built</h2>\
                <div>Please run <pre>python -m {api.state.app_name}.webview.build_client</pre></div>",
            content_type="text/html",
            status=404,
        )
    return web.FileResponse(client / "dist" / "index.html")


routes = app = sio = None


def init_web_app():
    import socketio
    from aiohttp import web

    global routes, app, sio

    routes = web.RouteTableDef()
    app = web.Application()
    app.router.add_get("/", index)
    # sio = socketio.AsyncServer(cors_allowed_origins="*", serializer='msgpack')
    sio = socketio.AsyncServer(cors_allowed_origins="*")
    sio.attach(app)

    def rest_get(fn):
        """
        Add a REST (GET) route for the function, which can also be used for
        """

        @routes.get(f"/{fn.__name__}")
        async def handler(request: web.Request):
            result = await fn(**request.query)
            return web.json_response(result)

        # pass the function to the next decorator unchanged...
        return fn

    def wrap_with_sid(function: Callable):
        """
        throw away first parameter sid: str
        for compatibility with socket.io
        (none of the API functions use sid value)
        """

        @functools.wraps(function)
        async def with_sid(sid: str, *args, **kwargs):
            return await function(*args, **kwargs)

        return with_sid

    api.EMITTERS["socketio"] = sio.emit
    for name, action in api.REGISTRY.items():
        sio.on(name, handler=wrap_with_sid(action))
        rest_get(action)

    @sio.event
    async def connect(sid: str, environ, data=None):
        for topic, contents in api.state.topics.items():
            message = contents[-1] if len(contents) > 0 else None
            if message is not None:
                await sio.emit(topic, message, to=sid)
        logger.info(f"Connected: session ID {sid}")

    @sio.event
    def disconnect(sid):
        logger.info(f"Disconnected: session ID {sid}")

    @sio.event
    async def set_base_path(sid: str, pathlist: List[str]):
        path = str(Path(*pathlist))
        persistent_settings.set_value("base_path", path, application=api.state.app_name)

    async def disconnect_all_clients():
        # disconnect all clients:
        clients = list(sio.manager.rooms.get("/", {None: {}}).get(None).keys())
        for client in clients:
            await sio.disconnect(client)
        while clients:
            clients = list(sio.manager.rooms.get("/", {None: {}}).get(None).keys())
            await asyncio.sleep(0.1)

    async def _shutdown():
        await disconnect_all_clients()
        logger.info("webserver shutdown tasks complete")
        await asyncio.sleep(0.1)
        raise web.GracefulExit()

    api._shutdown = _shutdown


def enable_convergence_kernel_heartbeat():
    from comm import create_comm

    comm = create_comm(target_name="heartbeat")

    async def send_heartbeat_on_convergence(event: str, *args, **kwargs):
        if event == "updated_convergence":
            comm.send({"status": "alive"})

    api.EMITTERS["convergence_heartbeat"] = send_heartbeat_on_convergence


def _create_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return sock


def _bind_random_port(hostname: str, preferred_port: int = PREFERRED_PORT):
    """
    Bind a random port to the given hostname.
    First tries the preferred port, then falls back to a random port.
    Returns the socket.
    """
    try:
        sock = _create_socket()
        sock.bind((hostname, preferred_port))
    except socket.error:
        sock.close()  # cleanup socket that didn't bind
        # create a new socket:
        sock = _create_socket()
        sock.bind((hostname, 0))
    return sock


def setup_app(options: BumpsOptions, sock: Optional[socket.socket] = None):
    from aiohttp import web, ClientSession

    static_assets_path = api.state.client_path / "dist" / "assets"
    if static_assets_path.exists():
        app.router.add_static("/assets", static_assets_path)

    async def notice(message: str):
        logger.info(message)

    on_startup, on_complete = cli.interpret_fit_options(options)
    app.on_startup.extend(on_startup)
    app.on_cleanup.append(lambda App: notice("cleanup task"))
    app.on_shutdown.extend(on_complete)
    app.on_shutdown.append(lambda App: notice("shutdown task"))
    # not sure why, but have to call shutdown twice to get it to work:
    app.on_shutdown.append(lambda App: api.shutdown())
    app.on_shutdown.append(lambda App: notice("shutdown complete"))

    # set initial path to cwd:
    app.add_routes(routes)
    hostname = "localhost" if not options.external else "0.0.0.0"

    if sock is None:
        if options.port == 0:
            sock = _bind_random_port(hostname)
        else:
            sock = _create_socket()
            sock.bind((hostname, options.port))

    host, port = sock.getsockname()
    api.state.hostname = host
    api.state.port = port

    if options.hub is not None:

        async def register_instance(application: web.Application):
            async with ClientSession() as client_session:
                await client_session.post(options.hub, json={"host": hostname, "port": port})

        app.on_startup.append(register_instance)

    if not options.headless:
        import webbrowser

        async def open_browser(app: web.Application):
            loop = asyncio.get_running_loop()
            loop.call_later(0.5, lambda: webbrowser.open_new_tab(f"http://{hostname}:{port}/"))

        app.on_startup.append(open_browser)

    if options.convergence_heartbeat:
        enable_convergence_kernel_heartbeat()

    if TRACE_MEMORY:
        import tracemalloc

        tracemalloc.start()

    return sock


def start_from_cli(options: BumpsOptions):
    from aiohttp import web

    init_web_app()
    runsock = setup_app(options=options, sock=None)
    web.run_app(app, sock=runsock)


server_task = None


def bumps_server():
    global server_task

    # Start the server
    server_task = asyncio.create_task(start_app(jupyter_link=True))
    return server_task


async def start_app(
    options: BumpsOptions,
    sock: socket.socket = None,
    jupyter_link: bool = False,
    jupyter_heartbeat: bool = False,
):
    from aiohttp import web

    init_web_app()

    # this function is called from jupyter notebook, so set headless = True
    options.headless = True
    # TODO: redirect logging somewhere perhaps
    # from .logger import setup_client_handler
    # setup_client_handler(logging.INFO, action=lambda msg: msg)
    runsock = setup_app(options=options, sock=sock)
    runner = web.AppRunner(app, handle_signals=False)
    await runner.setup()
    site = web.SockSite(runner, sock=runsock)
    await site.start()

    if jupyter_heartbeat:
        enable_convergence_kernel_heartbeat()

    if jupyter_link:
        return open_tab_link()
    else:
        url = get_server_url()
        print(f"webserver started: {url}")


def create_server_task():
    return asyncio.create_task(start_app())


def get_server_url():
    port = getattr(api.state, "port", None)
    if port is None:
        raise ValueError("The web server has not been started.")

    # detect if running through Jupyter Hub
    if "JUPYTERHUB_SERVICE_PREFIX" in os.environ:
        url = f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}/proxy/{port}/"
    elif api.state.hostname in ("localhost", "127.0.0.1"):  # local server
        url = f"http://{api.state.hostname}:{port}/"
    else:  # external server, e.g. TACC
        url = f"/proxy/{port}/"
    return url


def display_inline_jupyter(width: Union[str, int] = "100%", height: Union[str, int] = 600, single_panel=None) -> None:
    """
    Display the web server in an iframe.

    This is useful for displaying the web server in a Jupyter notebook.

    :param width: The width of the iframe.
    :param height: The height of the iframe.
    """
    from IPython.display import display, IFrame

    url = get_server_url()
    kwargs = dict(single_panel=single_panel) if single_panel is not None else {}
    display(
        IFrame(
            src=url,
            width=width,
            height=height,
            extras=['style="resize: both;"'],
            **kwargs,
        )
    )


def open_tab_link(single_panel=None) -> None:
    """
    Open the web server in a new tab in the default web browser.
    """
    from IPython.display import display, HTML

    url = get_server_url()
    if single_panel is not None:
        url += f"?single_panel={single_panel}"
    src = f'<h3><a href="{url}" target="_blank">Open Webview in Tab</a></h3>'
    display(HTML(src))
