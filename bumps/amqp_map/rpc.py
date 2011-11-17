import threading
from pickle import loads, dumps

from amqplib import client_0_8 as amqp

from .threaded import daemon

# TODO: create rpc exchange to avoid accidental collisions from wrong names

class RPCMixin:
    """
    Add RPC capabilities to a class
    """
    class RemoteException(Exception): pass
    def rpc_init(self, server, service="", provides=None):
        channel = server.channel()
        queue,_,_ = channel.queue_declare(queue=service,
                                          durable=False,
                                          exclusive=True,
                                          auto_delete=True)
        channel.queue_bind(queue=queue,
                           exchange="amq.direct",
                           routing_key=queue)
        channel.basic_consume(queue=queue,
                              callback=self._rpc_process,
                              no_ack=False)
        self._rpc_queue = queue
        self._rpc_channel = channel
        self._rpc_id = 0
        self._rpc_provides = provides
        self._rpc_sync = threading.Condition()
        self._rpc_results = {}
    @daemon
    def rpc_daemon(self):
        self.rpc_serve()
    def rpc_serve(self):
        while True:
            #print "waiting on channel"
            self._rpc_channel.wait()
        self._rpc_channel.close()
    def rpc(self, service, method, *args, **kw):
        self._rpc_send_call(service, ("call", method, args, kw))
        return self.rpc_wait(str(self._rpc_id))

    def rpc_async(self, service, method, *args, **kw):
        self._rpc_send_call(service, ("call", method, args, kw))
        return lambda: self.rpc_wait(str(self._rpc_id))

    def rpc_send(self, service, method, *args, **kw):
        self._rpc_send_call(service, ("send", method, args, kw))
        return str(self._rpc_id)

    def rpc_wait(self, rpc_id):
        # TODO: add timeout
        while rpc_id not in self._rpc_results:
            #print "wait results",self._rpc_results
            self._rpc_sync.acquire()
            self._rpc_sync.wait()
            self._rpc_sync.release()
        result = self._rpc_results.pop(rpc_id)
        if isinstance(result,Exception):
            raise result
        return result

    # Send messages
    def _rpc_send_call(self, service, parts):
        self._rpc_id += 1
        msg = amqp.Message(body=dumps(parts),
                           reply_to=self._rpc_queue,
                           message_id = str(self._rpc_id))
        self._rpc_channel.basic_publish(msg,
                                        exchange="amq.direct",
                                        routing_key=service)
    def _rpc_send_response(self, msg, result):
        #print "responding to",msg.reply_to,msg.message_id,"with",result
        resp = amqp.Message(body=dumps(("response",result)),
                           message_id=msg.message_id)
        self._rpc_channel.basic_publish(resp,
                                        exchange="amq.direct",
                                        routing_key=msg.reply_to)
    def _rpc_send_error(self, msg, str):
        resp = amqp.Message(body=dumps(("error",str)),
                           message_id=msg.message_id)
        self._rpc_channel.basic_publish(resp,
                                        exchange="amq.direct",
                                        routing_key=msg.reply_to)

    # Receive messages
    def _rpc_process(self, msg):
        try:
            parts = loads(msg.body)
            #print "process",parts
            # TODO: how do you use message headers properly?
            if parts[0] == "send":
                self._rpc_recv_send(msg, *parts[1:])
            elif parts[0] == "call":
                self._rpc_recv_call(msg, *parts[1:])
            elif parts[0] == "response":
                self._rpc_recv_response(msg, *parts[1:])
            elif parts[0] == "error":
                self._rpc_recv_error(msg, *parts[1:])
        except:
            raise
            return self._rpc_send_error(msg, "Invalid message")

    def _rpc_recv_call(self, msg, method, args, kw):
        if not self._rpc_valid_method(method):
            return self._rpc_send_error(msg, "Invalid method")
        fn = getattr(self, method)
        try:
            result = fn(*args, **kw)
        except:
            return self._rpc_error(msg, "Invalid arguments")
        return self._rpc_send_response(msg, result)

    def _rpc_recv_send(self, msg, method, args, kw):
        # TODO: silently ignore errors?
        if not self._rpc_valid_method(method):
            return
        fn = getattr(self, method)
        try:
            fn(*args, **kw)
        except:
            pass

    def _rpc_recv_response(self, msg, result):
        self._rpc_results[msg.message_id] = result
        self._rpc_sync.acquire()
        self._rpc_sync.notify()
        self._rpc_sync.release()

    def _rpc_recv_error(self, msg, str):
        self._rcp_results[msg.message_id] = self.RemoteException(str)
        self._rpc_sync.acquire()
        self._rpc_sync.notify()
        self._rpc_sync.release()

    def _rpc_valid_method(self, method):
        if self._rpc_provides: return method in self._rpc_provides
        if method.startswith('_') or method.startswith('rpc'): return False
        if not hasattr(self, method): return False
        return callable(getattr(self,method))

class RPC(object):
    """
    Connection to an rpc server as AMQP exchange.  Attributes are the
    names of rpc service queues.  Accessing an attribute creates a proxy.
    """
    def __init__(self, server):
        self._rpc = RPCMixin()
        self._rpc.rpc_init(server)
        self._rpc.rpc_daemon()
    def __getattr__(self, service):
        return RPCProxy(self._rpc, service)
class RPCProxy(object):
    """
    Proxy to an AMQP exchange rpc service.
    """
    def __init__(self, connection, service):
        self._rpc = connection
        self._service = service
    def __getattr__(self, method):
        return lambda *args, **kw: self._rpc.rpc(self._service, method, *args, **kw)
