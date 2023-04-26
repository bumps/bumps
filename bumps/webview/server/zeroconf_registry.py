import asyncio
import logging
from typing import Any, Dict, Optional, cast
from zeroconf import IPVersion
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf
import datetime

DEBUG = False


class LocalRegistry:
    def __init__(self, ip_version: IPVersion = IPVersion.All, type_ : str = "_bumps._tcp.local."):
        self.interfaces = ["127.0.0.1"]
        self.ip_version = ip_version
        self.aiozc = AsyncZeroconf(ip_version=ip_version, interfaces=["127.0.0.1"])
        self.type_ = type_

    def register(self, port: int, name: str, properties: Optional[Dict[str, str]]=None):
        properties = properties if properties is not None else {}
        properties.update({"start_time": datetime.datetime.now().timestamp()})
        info = AsyncServiceInfo(
            self.type_,
            f"{name} (port: {port}).{self.type_}",
            addresses=["127.0.0.1"],
            port=port,
            host_ttl = 5,
            other_ttl = 10,
            properties = {"start_time": datetime.datetime.now().timestamp()}
            # properties=properties
        )
        return self.aiozc.async_register_service(info)

    def unregister(self, port: int, name: str, properties: Optional[Dict[str, str]]=None):
        properties = properties if properties is not None else {}
        properties.update({"start_time": datetime.datetime.now().timestamp()})
        info = AsyncServiceInfo(
            self.type_,
            f"{name} ({port}).{self.type_}",
            addresses=["127.0.0.1"],
            port=port,
            properties={},
            host_ttl = 5,
            other_ttl = 10,
            # properties=properties
        )
        return self.aiozc.async_unregister_service(info)

    def close(self):
        return self.aiozc.async_unregister_all_services()


# adapted from https://github.com/python-zeroconf/python-zeroconf/blob/master/examples/async_registration.py
from zeroconf import IPVersion, ServiceStateChange, Zeroconf
from zeroconf.asyncio import (
    AsyncServiceBrowser,
    AsyncServiceInfo,
    AsyncZeroconf,
    AsyncZeroconfServiceTypes,
)


class AsyncRunner:
    def __init__(self, sio, service="_bumps._tcp.local.") -> None:
        self.aiobrowser: Optional[AsyncServiceBrowser] = None
        self.aiozc: Optional[AsyncZeroconf] = None
        self.sio = sio
        self.service = service
        self.servers = {}

    def on_service_state_change(self, 
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if DEBUG:
            print(f"Service {name} of type {service_type} state changed: {state_change}")
        asyncio.ensure_future(self._handle_state_change(zeroconf, service_type, name, state_change))

    async def _handle_state_change(self, zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zeroconf, 3000)
        stripped_name = name.replace(f".{service_type}", '')
        
        if DEBUG:
            print("Info from zeroconf.get_service_info: %r\n" % (info))
        serializable_properties = dict((k.decode(), v.decode()) for k,v in info.properties.items())
        if info:
            if state_change is ServiceStateChange.Added:
                addresses = ["%s:%d" % (addr, cast(int, info.port)) for addr in info.parsed_scoped_addresses()]
                start_time = float(info.properties.get(b"start_time", b"0").decode()) * 1000 # in milliseconds
                self.servers[stripped_name] = {"addresses": addresses, "start_time": start_time}
            elif state_change is ServiceStateChange.Removed:
                del self.servers[stripped_name]

            await self.sio.emit("update")
        else:
            # no info
            if DEBUG:
                print("  No info\n")

    def get_servers(self, sid=""):
        return self.servers

    async def async_run(self) -> None:
        self.aiozc = AsyncZeroconf(ip_version=IPVersion.All, interfaces=["127.0.0.1"])

        services = [self.service]
        self.aiobrowser = AsyncServiceBrowser(
            self.aiozc.zeroconf, services, handlers=[self.on_service_state_change]
        )

    async def async_close(self) -> None:
        assert self.aiozc is not None
        assert self.aiobrowser is not None
        await self.aiobrowser.async_cancel()
        await self.aiozc.async_close()
