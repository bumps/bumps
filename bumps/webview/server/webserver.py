# from .main import setup_bumps
from dataclasses import dataclass
import functools
import os
import signal
import socket
from typing import Callable, Dict, Optional
import warnings
from aiohttp import web, ClientSession
import asyncio
import socketio
from typing import Union, List
from pathlib import Path
import json
import re
import sys

import matplotlib
matplotlib.use("agg")

import mimetypes
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

from . import api
from .fit_thread import EVT_FIT_PROGRESS
from .state_hdf5_backed import SERIALIZERS, UNDEFINED
from .logger import logger, list_handler, console_handler
from . import persistent_settings

TRACE_MEMORY = False
CDN_TEMPLATE = "https://cdn.jsdelivr.net/npm/bumps-webview-client@{client_version}/dist/{client_version}"

# can get by name and not just by id

routes = web.RouteTableDef()
# sio = socketio.AsyncServer(cors_allowed_origins="*", serializer='msgpack')
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
CLIENT_PATH = Path(__file__).parent.parent / 'client'
APPLICATION_NAME = "bumps"

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

    
@sio.event
async def connect(sid: str, environ, data=None):
    for topic, contents in api.state.topics.items():
        message = contents[-1] if len(contents) > 0 else None
        if message is not None:
            await sio.emit(topic, message, to=sid)
    logger.info(f"connect {sid}")

@sio.event
def disconnect(sid):
    logger.info(f"disconnect {sid}")

@sio.event
async def set_base_path(sid: str, pathlist: List[str]):
    path = str(Path(*pathlist))
    persistent_settings.set_value("base_path", path, application=APPLICATION_NAME)

async def disconnect_all_clients():
    # disconnect all clients:
    clients = list(sio.manager.rooms.get('/', {None: {}}).get(None).keys())
    for client in clients:
        await sio.disconnect(client)
    while clients:
        clients = list(sio.manager.rooms.get('/', {None: {}}).get(None).keys())
        await asyncio.sleep(0.1)

async def _shutdown():
    await disconnect_all_clients()
    logger.info("webserver shutdown tasks complete")
    await asyncio.sleep(0.1)
    raise web.GracefulExit()

api._shutdown = _shutdown

import argparse

@dataclass
class BumpsOptions:
    """ provide type hints for arguments """
    filename: Optional[str] = None
    headless: bool = True
    external: bool = False
    port: int = 0
    hub: Optional[str] = None
    fit: Optional[str] = None
    start: bool = False
    read_store: Optional[str] = None
    write_store: Optional[str] = None
    store: Optional[str] = None
    exit: bool = False
    serializer: SERIALIZERS = "dill"
    trace: bool = False
    parallel: int = 0
    path: Optional[str] = None
    no_auto_history: bool = False

OPTIONS_CLASS = BumpsOptions

def get_commandline_options(arg_defaults: Optional[Dict]=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', help='problem file to load, .py or .json (serialized) fitproblem')
    # parser.add_argument('-d', '--debug', action='store_true', help='autoload modules on change')
    parser.add_argument('-x', '--headless', action='store_true', help='do not automatically load client in browser')
    parser.add_argument('--external', action='store_true', help='listen on all interfaces, including external (local connections only if not set)')
    parser.add_argument('-p', '--port', default=0, type=int, help='port on which to start the server')
    parser.add_argument('--hub', default=None, type=str, help='api address of parent hub (only used when called as subprocess)')
    parser.add_argument('--fit', default=None, type=str, choices=list(api.FITTERS_BY_ID.keys()), help='fitting engine to use; see manual for details')
    parser.add_argument('--start', action='store_true', help='start fit when problem loaded')
    parser.add_argument('--store', default=None, type=str, help='set read_store and write_store to same file')
    parser.add_argument('--read_store', default=None, type=str, help='read initial session state from file (overrides --store)')
    parser.add_argument('--write_store', default=None, type=str, help='output file for session state (overrides --store)')
    parser.add_argument('--exit', action='store_true', help='end process when fit complete (fit results lost unless write_store is specified)')
    parser.add_argument('--serializer', default=OPTIONS_CLASS.serializer, type=str, choices=["pickle", "dill", "dataclass"], help='strategy for serializing problem, will use value from store if it has already been defined')
    parser.add_argument('--trace', action='store_true', help='enable memory tracing (prints after every uncertainty update in dream)')
    parser.add_argument('--parallel', default=0, type=int, help='run fit using multiprocessing for parallelism; use --parallel=0 for all cpus')
    parser.add_argument('--path', default=None, type=str, help='set initial path for save and load dialogs')
    parser.add_argument('--no_auto_history', action='store_true', help='disable auto-appending problem state to history on load and at fit end')
    # parser.add_argument('-c', '--config-file', type=str, help='path to JSON configuration to load')
    namespace = OPTIONS_CLASS()
    if arg_defaults is not None:
        logger.debug(f'arg_defaults: {arg_defaults}')
        for k,v in arg_defaults.items():
            setattr(namespace, k, v)
    args = parser.parse_args(namespace=namespace)
    return args

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

def setup_sio_api():
    api.emit = sio.emit
    for (name, action) in api.REGISTRY.items():
        sio.on(name, handler=wrap_with_sid(action))
        rest_get(action)

def setup_app(sock: Optional[socket.socket] = None, options: OPTIONS_CLASS = OPTIONS_CLASS()):
    # check if the locally-build site has the correct version:
    with open(CLIENT_PATH / 'package.json', 'r') as package_json:
        client_version = json.load(package_json)['version'].strip()

    static_assets_path = CLIENT_PATH / 'dist' / client_version / 'assets'

    if Path.exists(static_assets_path):
        app.router.add_static('/assets', static_assets_path)

    async def index(request):
        """Serve the client-side application."""
        local_client_path = CLIENT_PATH / 'dist' / client_version

        if local_client_path.is_dir():
            return web.FileResponse(local_client_path / 'index.html')
        else:
            CDN = CDN_TEMPLATE.format(client_version=client_version)
            with open(CLIENT_PATH / 'index_template.txt', 'r') as index_template:
                index_html = index_template.read().format(cdn=CDN)
            return web.Response(body=index_html, content_type="text/html")
        
    app.router.add_get('/', index)

    api.state.base_path = persistent_settings.get_value('base_path', str(Path().absolute()), application=APPLICATION_NAME)
    if options.path is not None:
        if Path(options.path).exists():
            api.state.base_path = options.path
        else:
            logger.warning(f"specified path {options.path} does not exist, reverting to current directory")

    if options.read_store is not None and options.store is not None:
        warnings.warn("read_store and store are both set; read_store will be used to initialize state")
    if options.write_store is not None and options.store is not None:
        warnings.warn("write_store and store are both set; write_store will be used to save state")

    read_store = options.read_store if options.read_store is not None else options.store
    write_store = options.write_store if options.write_store is not None else options.store
    if read_store is not None:
        read_store_path = Path(read_store).absolute()
        api.state.read_session_file(str(read_store_path))
        if write_store is None:
            api.state.shared.session_output_file = dict(pathlist=list(read_store_path.parent.parts), filename=read_store_path.name)
    if write_store is not None:
        write_store_path = Path(write_store).absolute()
        api.state.shared.session_output_file = dict(pathlist=list(write_store_path.parent.parts), filename=write_store_path.name)
        api.state.shared.autosave_session = True

    if api.state.problem.serializer is None or api.state.problem.serializer == "":
        api.state.problem.serializer = options.serializer

    if options.no_auto_history:
        api.state.shared.autosave_history = False

    if options.trace:
        global TRACE_MEMORY
        TRACE_MEMORY = True
        api.TRACE_MEMORY = True

    # app.on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    if options.fit is not None:
        app.on_startup.append(lambda App: api.state.shared.set('selected_fitter', options.fit))

    fitter_id = options.fit
    if fitter_id is None:
        fitter_id = api.state.shared.selected_fitter
    if fitter_id is None or fitter_id is UNDEFINED:
        fitter_id = 'amoeba'
    fitter_settings = api.FITTER_DEFAULTS[fitter_id]

    api.state.parallel = options.parallel

    # if args.steps is not None:
    #     fitter_settings["steps"] = args.steps

    if options.filename is not None:
        filepath = Path(options.filename).absolute()
        pathlist = list(filepath.parent.parts)
        filename = filepath.name
        logger.debug(f"fitter for filename {filename} is {fitter_id}")
        async def load_problem(App=None):
            await api.load_problem_file(pathlist, filename)
        
        app.on_startup.append(load_problem)

    if options.start:
        async def start_fit(App=None):
            if api.state.problem is not None:
                await api.start_fit_thread(fitter_id, fitter_settings["settings"], options.exit)
        app.on_startup.append(start_fit)
    else:
        # signal that no fit is running at startup, even if a fit was
        # interrupted and the state was saved:
        app.on_startup.append(lambda App: api.state.shared.set('active_fit', {}))

    async def notice(message: str):
        logger.info(message)
    app.on_cleanup.append(lambda App: notice("cleanup task"))
    app.on_shutdown.append(lambda App: notice("shutdown task"))
    # not sure why, but have to call shutdown twice to get it to work:
    app.on_shutdown.append(lambda App: api.shutdown())
    app.on_shutdown.append(lambda App: notice("shutdown complete"))

    # set initial path to cwd:
    app.add_routes(routes)
    hostname = 'localhost' if not options.external else '0.0.0.0'

    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
            loop = asyncio.get_event_loop()
            loop.call_later(0.5, lambda: webbrowser.open_new_tab(f"http://{hostname}:{port}/"))
        app.on_startup.append(open_browser)

    if TRACE_MEMORY:
        import tracemalloc
        tracemalloc.start()

    return sock

def main(options: Optional[OPTIONS_CLASS] = None, sock: Optional[socket.socket] = None):
    # this entrypoint will be used to start gui, so set headless = False
    # (other contexts e.g. jupyter notebook will directly call start_app)
    logger.addHandler(console_handler)
    options = get_commandline_options(arg_defaults={"headless": False}) if options is None else options
    logger.info(dict(options=options))
    setup_sio_api()
    runsock = setup_app(options=options, sock=None)
    web.run_app(app, sock=runsock)

async def start_app(options: OPTIONS_CLASS = OPTIONS_CLASS(), sock: socket.socket = None, jupyter_link: bool = False):
    # this function is called from jupyter notebook, so set headless = True
    options.headless = True
    # redirect logging to a list
    logger.addHandler(list_handler)
    setup_sio_api()
    runsock = setup_app(options=options, sock=sock)
    runner = web.AppRunner(app, handle_signals=False)
    await runner.setup()
    site = web.SockSite(runner, sock=runsock)
    await site.start()
    if jupyter_link:
        return open_tab_link()
    else:
        url = get_server_url()
        print(f"webserver started: {url}")

def create_server_task():
    return asyncio.create_task(start_app())


def get_server_url():
    from bumps.webview.server import api

    port = getattr(api.state, 'port', None)
    if port is None:
        raise ValueError("The web server has not been started.")

    # detect if running through Jupyter Hub
    if 'JUPYTERHUB_SERVICE_PREFIX' in os.environ:
        url = f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}/proxy/{port}/"
    elif api.state.hostname == 'localhost': # local server
        url = f"http://{api.state.hostname}:{port}/"
    else: # external server, e.g. TACC
        url = f"/proxy/{port}/"
    return url

def display_inline_jupyter(width: Union[str,int]="100%", height: Union[str, int]=600, single_panel=None) -> None:
    """
    Display the web server in an iframe.

    This is useful for displaying the web server in a Jupyter notebook.

    :param width: The width of the iframe.
    :param height: The height of the iframe.
    """
    from IPython.display import display, IFrame

    url = get_server_url()
    kwargs = dict(single_panel=single_panel) if single_panel is not None else {}
    display(IFrame(src=url, width=width, height=height, extras=['style="resize: both;"'], **kwargs))

def open_tab_link(single_panel=None) -> None:
    """
    Open the web server in a new tab in the default web browser.
    """
    from IPython.display import Javascript, display, HTML

    url = get_server_url()
    if single_panel is not None:
        url += f"?single_panel={single_panel}"
    src = f'<h3><a href="{url}" target="_blank">Open Webview in Tab</a></h3>'
    display(HTML(src))

if __name__ == '__main__':
    main()
