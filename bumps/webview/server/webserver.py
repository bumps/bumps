# from .main import setup_bumps
from dataclasses import dataclass
import signal
import socket
from typing import Callable, Dict, Optional
from aiohttp import web, ClientSession
import asyncio
import socketio
from pathlib import Path
import json

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
from .state_hdf5_backed import SERIALIZERS

TRACE_MEMORY = False

# can get by name and not just by id

routes = web.RouteTableDef()
# sio = socketio.AsyncServer(cors_allowed_origins="*", serializer='msgpack')
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
CLIENT_PATH = Path(__file__).parent.parent / 'client'

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


    try:
        local_version = open(index_path / 'VERSION', 'rt').read().strip()
    except FileNotFoundError:
        local_version = None

    print(index_path, local_version, client_version, local_version == client_version)
    if client_version == local_version:
        return web.FileResponse(index_path / 'index.html')
    else:
        CDN = f"https://cdn.jsdelivr.net/npm/bumps-webview-client@{client_version}/dist"
        with open(client_path / 'index_template.txt', 'r') as index_template:
            index_html = index_template.read().format(cdn=CDN)
        return web.Response(body=index_html, content_type="text/html")
    
@api.register
async def connect(sid, environ, data=None):
    # re-send last message for all topics
    # now that panels are retrieving topics when they load, is this still
    # needed or useful?
    for topic, contents in api.state.topics.items():
        message = contents[-1] if len(contents) > 0 else None
        if message is not None:
            await api.emit(topic, message, to=sid)
    print("connect ", sid)

@api.register
def disconnect(sid):
    print('disconnect ', sid)

async def disconnect_all_clients():
    # disconnect all clients:
    clients = list(sio.manager.rooms.get('/', {None: {}}).get(None).keys())
    for client in clients:
        await sio.disconnect(client)

async def _shutdown():
    raise web.GracefulExit()

api._shutdown = _shutdown

app["shutdown"] = lambda: asyncio.create_task(api.shutdown())

import argparse

@dataclass
class Options:
    """ provide type hints for arguments """
    filename: Optional[str] = None
    headless: bool = False
    external: bool = False
    port: int = 0
    hub: Optional[str] = None
    fit: Optional[str] = None
    start: bool = False
    store: Optional[str] = None
    exit: bool = False
    serializer: SERIALIZERS = "dill"
    trace: bool = False


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
    parser.add_argument('--store', default=None, type=str, help='backing file for state')
    parser.add_argument('--exit', action='store_true', help='end process when fit complete (fit results lost unless store is specified)')
    parser.add_argument('--serializer', default='dill', type=str, choices=["pickle", "dill", "dataclass"], help='strategy for serializing problem, will use value from store if it has already been defined')
    parser.add_argument('--trace', action='store_true', help='enable memory tracing (prints after every uncertainty update in dream)')
    # parser.add_argument('-c', '--config-file', type=str, help='path to JSON configuration to load')
    if arg_defaults is not None:
        parser.set_defaults(**arg_defaults)
    args = parser.parse_args(namespace=Options())
    return args


def setup_sio_api():
    api.app = app
    api.emit = sio.emit
    for (name, action) in api.REGISTRY.items():
        sio.on(name, handler=action)

def setup_app(index: Callable=index, static_assets_path: Path=static_assets_path, sock: Optional[socket.socket] = None, options: Options = Options()):
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

    if options.store is not None:
        api.state.setup_backing(session_file_name=options.store, in_memory=False)

    if api.state.problem.serializer is None:
        api.state.problem.serializer = options.serializer

    if options.trace:
        global TRACE_MEMORY
        TRACE_MEMORY = True
        api.TRACE_MEMORY = True

    # app.on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    if options.fit is not None:
        app.on_startup.append(lambda App: api.publish('', 'fitter_active', options.fit))

    fitter_id = options.fit
    if fitter_id is None:
        fitter_active_topic = api.state.topics["fitter_active"]
        if len(fitter_active_topic) > 0:
            fitter_id = fitter_active_topic[-1]["message"]
    if fitter_id is None:
        fitter_id = 'amoeba'
    fitter_settings = api.FITTER_DEFAULTS[fitter_id]

    # if args.steps is not None:
    #     fitter_settings["steps"] = args.steps

    if options.filename is not None:
        filepath = Path(options.filename)
        pathlist = list(filepath.parent.parts)
        filename = filepath.name
        start = options.start
        print(f"fitter for filename {filename} is {fitter_id}")
        async def startup_task(App=None):
            await api.load_problem_file("", pathlist, filename)
            if start:
                await api.start_fit_thread("", fitter_id, fitter_settings["settings"], options.exit)
        
        app.on_startup.append(startup_task)

        # app.on_startup.append()

    # app.on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    async def add_signal_handler(app):
        try:
            app.loop.add_signal_handler(signal.SIGINT, app["shutdown"])
        except NotImplementedError:
            # Windows does not implement this method, but still handles KeyboardInterrupt
            pass

    app.on_startup.append(lambda App: add_signal_handler(App))
    async def notice(message: str):
        print(message)
    app.on_cleanup.append(lambda App: notice("cleanup task"))
    app.on_shutdown.append(lambda App: notice("shutdown task"))
    app.on_shutdown.append(lambda App: api.stop_fit())
    app.on_shutdown.append(lambda App: api.state.cleanup())
    app.on_shutdown.append(lambda App: notice("shutdown complete"))

    # set initial path to cwd:
    api.state.problem.pathlist = list(Path().absolute().parts)
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

def main(options: Optional[Options] = None, sock: Optional[socket.socket] = None):
    options = get_commandline_options() if options is None else options
    try:
        asyncio.run(start_app(options, sock))
    except KeyboardInterrupt:
        print("stopped by KeyboardInterrupt.")

async def start_app(options: Options, sock: socket.socket):
    setup_sio_api()
    runsock = setup_app(options=options, sock=sock)
    await web._run_app(app, sock=runsock)

if __name__ == '__main__':
    main()
