#!/usr/bin/env python
import time

from amqp_map.config import SERVICE_HOST
from amqp_map.core import connect
from amqp_map.rpc import RPC

rpc = RPC(connect(SERVICE_HOST))
for item in sys.argv[1:]:
    print rpc.echo.echo(item)
