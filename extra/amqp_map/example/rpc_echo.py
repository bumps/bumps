import time

from amqp_map.core import connect
from amqp_map.config import SERVICE_HOST
from amqp_map.rpc import RPCMixin

server = connect(SERVICE_HOST)

class Echo(object,RPCMixin):
    def __init__(self, server):
        self.rpc_init(server, service="echo")
    def echo(self, msg):
        return msg

service = Echo(server)
service.rpc_serve()
