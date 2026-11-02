"""
Check that the socket.io event names used by the webview client match the server.

The client calls server handlers and listens for server events by name, through
an untyped socket, so a rename on either side is invisible to the type checker
and to the client unit tests (which use a fake socket). These tests scan the
client source for event names and compare them with what the server registers.
"""

import re
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

CLIENT_SRC = Path(__file__).parent / "client" / "src"

# socket.io handlers defined directly in webserver.py rather than via api.register
SOCKETIO_HANDLERS = {"set_base_path"}

# Events the client listens for that are not shared-state fields or topics.
OTHER_SERVER_EVENTS = {
    "connect",  # socket.io built-in
    "disconnect",  # socket.io built-in
    "add_notification",  # api.add_notification
    "cancel_notification",  # api.add_notification
    "plotly_click",  # Plotly DOM event, not a socket event
    # No known sender in bumps or refl1d; kept for host applications.
    "base_path",  # FileBrowser.vue, added in 9660eb6
    "server_startup_status",  # ServerStartup.vue, added in 79f195a
}


def _client_names(pattern):
    if not CLIENT_SRC.exists():
        pytest.skip("webview client source not available")
    text = "\n".join(p.read_text(encoding="utf-8") for p in CLIENT_SRC.rglob("*") if p.suffix in {".ts", ".vue"})
    return set(re.findall(pattern, text))


def test_client_calls_exist_on_server():
    from bumps import api
    from bumps.webview import webserver  # noqa: F401 -- registers the server handlers

    called = _client_names(r"""asyncEmit\(\s*["'`](\w+)["'`]""")
    assert called, "found no asyncEmit calls; has the client API changed?"
    missing = called - set(api.REGISTRY) - SOCKETIO_HANDLERS
    assert not missing, f"client calls handlers the server does not define: {sorted(missing)}"


def test_client_listens_for_events_the_server_sends():
    from bumps.state import SharedState, TopicNameType

    listened = _client_names(r"""\.on\(\s*["'](\w+)["']""") | _client_names(r"""setupDrawLoop\(\s*["'](\w+)["']""")
    assert listened, "found no event listeners; has the client API changed?"
    # every shared-state change is broadcast under the field name (api.notify_shared_setting)
    shared = {f.name for f in fields(SharedState) if not f.name.startswith("_")}
    topics = set(get_args(TopicNameType))
    unknown = listened - shared - topics - OTHER_SERVER_EVENTS
    assert not unknown, f"client listens for events the server never sends: {sorted(unknown)}"
