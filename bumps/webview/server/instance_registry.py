

import argparse
import socketio
from aiohttp import web, ClientSession
from pathlib import Path
import socket
import mimetypes
from .zeroconf_registry import AsyncRunner

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

async def index(request):
    """Serve the client-side application."""
    # check if the locally-build site has the correct version:
    return web.FileResponse(index_path / 'instance_browser.html')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--headless', action='store_true', help='do not automatically load client in browser')
    parser.add_argument('--external', action='store_true', help='listen on all interfaces, including external (local connections only if not set)')
    parser.add_argument('-p', '--port', default=0, type=int, help='port on which to start the server')
    args = parser.parse_args()

    app.router.add_get('/', index)

    hostname = 'localhost' if not args.external else '0.0.0.0'

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((hostname, args.port))
    host, port = sock.getsockname()
    
    if not args.headless:
        import webbrowser
        async def open_browser(app: web.Application):
            app.loop.call_later(0.25, lambda: webbrowser.open_new_tab(f"http://{hostname}:{port}/"))
        app.on_startup.append(lambda App: open_browser(App))

    zeroconf_relay = AsyncRunner(sio)

    sio.on("get_servers", handler=zeroconf_relay.get_servers)
    app.on_startup.append(lambda App: zeroconf_relay.async_run())
    web.run_app(app, sock=sock)

if __name__ == '__main__':
    main()