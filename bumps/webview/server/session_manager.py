import asyncio
import argparse
import socketio
from aiohttp import web, ClientSession
from pathlib import Path
import socket
import mimetypes
from .zeroconf_registry import AsyncRunner, LocalRegistry

mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

routes = web.RouteTableDef()
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
index_path = Path(__file__).parent

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

async def index(request):
    """Serve the client-side application."""
    # check if the locally-build site has the correct version:
    return web.FileResponse(index_path / 'session_manager.html')

@sio.event
@rest_get
async def shutdown(sid: str=""):
    print("killing...")
    await sio.emit("server_shutting_down")
    shutdown_result = asyncio.gather(_shutdown(), return_exceptions=True)

async def _shutdown():
    raise web.GracefulExit()

def setup_app():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--headless', action='store_true', help='do not automatically load client in browser')
    parser.add_argument('--external', action='store_true', help='listen on all interfaces, including external (local connections only if not set)')
    parser.add_argument('-p', '--port', default=0, type=int, help='port on which to start the server')
    args = parser.parse_args()

    app.router.add_get('/', index)

    app.add_routes(routes)
    hostname = 'localhost' if not args.external else '0.0.0.0'

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((hostname, args.port))
    host, port = sock.getsockname()
    
    if not args.headless:
        import webbrowser
        async def open_browser(app: web.Application):
            loop = asyncio.get_event_loop()
            loop.call_later(0.5, lambda: webbrowser.open_new_tab(f"http://{hostname}:{port}/"))
        app.on_startup.append(lambda App: open_browser(App))

    registry = LocalRegistry()
    async def register_zeroconf(app: web.Application):
        return await registry.register(port, "Session Manager")
    app.on_startup.append(register_zeroconf)
    app.on_shutdown.append(lambda App: registry.close())

    zeroconf_relay = AsyncRunner(sio)

    sio.on("get_servers", handler=zeroconf_relay.get_servers)
    app.on_startup.append(lambda App: zeroconf_relay.async_run())

    return sock

def main():
    asyncio.run(start_app())

async def start_app():
    runsock = setup_app()
    await web._run_app(app, sock=runsock)

if __name__ == '__main__':
    main()