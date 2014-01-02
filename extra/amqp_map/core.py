
# Mechanisms for throttling the mapper when the function is expensive:
#
# 1) Do nothing.
#    PRO: no computation overhead
#    PRO: AMQP will use flow control when it runs low on memory
#    PRO: most maps are small
#    CON: may use excess memory on exchange
# 2) Use locks.
#    PRO: threading.Condition() makes this easy
#    PRO: good lock implementation means master consumes no resources
#    CON: may not handle keyboard interrupt correctly on some platforms
# 3) Use sleep.
#    PRO: simple implementation will work across platforms
#    CON: master stays in memory because it is restarted every 10 min.
#
# The current implementation uses locks to throttle.

## USE_LOCKS_TO_THROTTLE
import threading

## USE_SLEEP_TO_THROTTLE
#import time
#import os

#from dill import loads, dumps
from pickle import loads, dumps
import sys

from amqplib import client_0_8 as amqp #@UnresolvedImport if amqp isn't available

from . import config
from .url import URL
from .threaded import daemon

def connect(url, insist=False):
    url = URL(url, host="localhost", port=5672,
              user="guest", password="guest", path="/")
    host = ":".join( (url.host, str(url.port)) )
    userid,password = url.user,url.password
    virtual_host = "/" + url.path
    server = amqp.Connection(host=host, userid=userid, password=password,
                             virtual_host=virtual_host, insist=insist)
    return server

def start_worker(server, mapid, work):
    """
    Client side driver of the map work.

    The model should already be loaded before calling this.
    """
    # Create the exchange and the worker queue
    channel = server.channel()
    exchange = "park.map"
    map_queue = ".".join(("map",mapid))
    channel.exchange_declare(exchange=exchange, type="direct",
                             durable=False, auto_delete=True)
    channel.queue_declare(queue=map_queue, durable=False,
                          exclusive=False, auto_delete=True)

    #me = os.getpid()
    #os.system("echo '%s' > /home/pkienzle/map.%d"%('starting',me))

    # Prefetch requires basic_ack, basic_qos and consume with ack
    def _process_work(msg):
        # Check for sentinel
        if msg.reply_to == "":
            channel.basic_cancel(consumer)
        body = loads(msg.body)
        # Acknowledge delivery of message
        #print "processing...",body['index'],body['value']; sys.stdout.flush()
        #os.system("echo 'processing %s' >> /home/pkienzle/map.%d"%(body['value'],me))
        try:
            result = work(body['value'])
        except Exception as _exc:
            #os.system("echo 'error %s' >> /home/pkienzle/map.%d"%(_exc,me))
            result = None
        #os.system("echo 'returning %s' >> /home/pkienzle/map.%d"%(result,me))
        #print "done"
        channel.basic_ack(msg.delivery_tag)
        reply = amqp.Message(dumps(dict(index=body['index'],result=result)))
        channel.basic_publish(reply, exchange=exchange,
                              routing_key=msg.reply_to)
    #channel.basic_qos(prefetch_size=0, prefetch_count=1, a_global=False)
    consumer = channel.basic_consume(queue=map_queue, callback=_process_work,
                                     no_ack=False)
    while True:
        channel.wait()

class Mapper(object):
    def __init__(self, server, mapid):
        # Create the exchange and the worker and reply queues
        channel = server.channel()
        exchange = "park.map"
        channel.exchange_declare(exchange=exchange, type="direct",
                                 durable=False, auto_delete=True)

        map_channel = channel
        map_queue = ".".join(("map",mapid))
        map_channel.queue_declare(queue=map_queue, durable=False,
                                  exclusive=False, auto_delete=True)
        map_channel.queue_bind(queue=map_queue, exchange="park.map",
                               routing_key = map_queue)

        reply_channel = server.channel()
        #reply_queue = ".".join(("reply",mapid)) # Fixed Queue name
        reply_queue = "" # Let amqp create a temporary queue for us
        reply_queue,_,_ = reply_channel.queue_declare(queue=reply_queue,
                                                      durable=False,
                                                      exclusive=True,
                                                      auto_delete=True)
        reply_channel.queue_bind(queue=reply_queue, exchange="park.map",
                                 routing_key = reply_queue)
        reply_channel.basic_consume(queue=reply_queue,
                                    callback=self._process_result,
                                    no_ack=True)
        self.exchange = exchange
        self.map_queue = map_queue
        self.map_channel = map_channel
        self.reply_queue = reply_queue
        self.reply_channel = reply_channel

        ## USE_LOCKS_TO_THROTTLE
        self._throttle = threading.Condition()

    def close(self):
        self.channel.close()
    def _process_result(self, msg):
        self._reply = loads(msg.body)
        #print "received result",self._reply['index'],self._reply['result']
    @daemon
    def _send_map(self, items):
        for i,v in enumerate(items):
            self.num_queued = i
            #print "queuing %d %s"%(i,v)

            ## USE_LOCKS_TO_THROTTLE
            if  self.num_queued - self.num_processed > config.MAX_QUEUE:
                #print "sleeping at %d in %d out"%(i,self.num_processed)
                self._throttle.acquire()
                self._throttle.wait()
                self._throttle.release()
                #print "waking at %d in %d out"%(i,self.num_processed)

            # USE_SLEEP_TO_THROTTLE
            #sleep_time = 0.2
            #while i - self.num_processed > config.MAX_QUEUE:
            #    #print "sleeping %g with in=%d out=%d"%(sleep_time,self.num_queued,self.num_processed)
            #    time.sleep(sleep_time)
            #    sleep_time = min(2*sleep_time, 600)

            body = dumps(dict(index=i,value=v))
            msg = amqp.Message(body, reply_to=self.reply_queue, delivery_mode=1)
            self.map_channel.basic_publish(msg, exchange=self.exchange,
                                           routing_key=self.map_queue)

    def cancel(self):
        """
        Stop a running map.
        """
        raise NotImplementedError()
        # Need to clear the queued items and notify async that no more results.
        # Messages in transit need to be ignored, which probably means tagging
        # each map header with a call number so that previous calls don't
        # get confused with current calls.
        msg = amqp.Message("", reply_to="", delivery_mode=1)
        self.map_channel.basic_publish(msg, exchange=self.exchange,
                                       routing_key=self.map_queue)

    def async(self, items):
        # TODO: we should be able to flag completion somehow so that the
        # whole list does not need to be formed.
        items = list(items) # make it indexable
        self.num_items = len(items)
        # Queue items in separate thread so we can start receiving results
        # before all items are even queued
        self.num_processed = 0
        publisher = self._send_map(items)
        recvd = set()
        while self.num_processed < self.num_items:
            try: del self._reply
            except: pass
            self.reply_channel.wait()
            try:
                idx = self._reply['index']
            except:
                sys.stdout.flush()
                raise RuntimeError("Reply not received")
            if idx in recvd: continue
            recvd.add(idx)
            result = self._reply['result']
            #print "received %d %g"%(idx,result)
            self.num_processed += 1

            ## USE_LOCKS_TO_THROTTLE
            if self.num_queued - self.num_processed < config.MAX_QUEUE - 10:
                # Ten at a time go through for slow processes
                self._throttle.acquire()
                self._throttle.notify()
                self._throttle.release()

            yield idx,result
        publisher.join()
    def __call__(self, items):
        result = list(self.async(items))
        result = list(sorted(result,lambda x,y: cmp(x[0],y[0])))
        return zip(*result)[1]
