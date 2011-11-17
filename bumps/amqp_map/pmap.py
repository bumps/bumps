raise NotImplementedError  # This code is still a work in progress

import threading
import dill as pickle
from .rpc import RPCMixin

def pickle_worker(server):
    """
    Client side driver of the map work.
    """
    # Create the exchange and the worker queue
    channel = server.channel()
    rpc_channel = server.channel()
    exchange = "park.map"
    map_queue = "map.pickle"
    channel.exchange_declare(exchange=exchange, type="direct",
                             durable=False, auto_delete=True)
    channel.queue_declare(queue=map_queue, durable=False,
                          exclusive=False, auto_delete=True)

    _rpc_queue,_,_ = channel.queue_declare(queue=service,
                                           durable=False,
                                           exclusive=True,
                                           auto_delete=True)
    channel.queue_bind(queue=_rpc_queue,
                       exchange="amq.direct",
                       routing_key=queue)


    _cache = {}
    def _fetch_function(queue, mapid):
        reply = amqp.Message(dumps(dict(mapid=mapid,
                                        sendfunction=rpc_queue)))
        channel.basic_publish(reply, exchange=exchange,
                              routing_key=queue)
        def _receive_function(msg):
            rpc_channel.basic_cancel(tag)
            body = pickle.loads(msg.body)
            _cache[body['mapid']] = pickle.loads(body['function'])

        tag = channel.basic_consume(queue=queue,
                                    callback=_receive_function,
                                    no_ack=False)
        rpc_channel.wait() # Wait for function id

    def _process_work(msg):
        # Check for sentinel
        if msg.reply_to == "": channel.basic_cancel(consumer)

        body = pickle.loads(msg.body)
        mapid = body['mapid']
        if mapid not in _cache:
            _fetch_function(msg.reply_to, mapid)
        function = _cache[mapid]
        if function == None:
            channel.basic_ack(msg.delivery_tag)
            return

        # Acknowledge delivery of message
        #print "processing...",body['index'],body['value']
        try:
            result = function(body['value'])
        except:
            result = None
        #print "done"
        channel.basic_ack(msg.delivery_tag)
        reply = dict(index=body['index'], result=result, mapid=mapid)
        replymsg = amqp.Message(pickle.dumps(reply))
        channel.basic_publish(replymsg, exchange=exchange,
                              routing_key=msg.reply_to)
    #channel.basic_qos(prefetch_size=0, prefetch_count=1, a_global=False)
    consumer = channel.basic_consume(queue=map_queue, callback=_process_work,
                                     no_ack=False)
    while True:
        channel.wait()



class PickleMapper(object, RPCMixin):
    def server(self, server):
        # Create the exchange and the worker and reply queues
        channel = server.channel()
        exchange = "park.map"
        channel.exchange_declare(exchange=exchange, type="direct",
                                 durable=False, auto_delete=True)

        map_channel = channel
        map_queue = "map.pickle"
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

        ## Start the rpc server
        self.rpc_init(server, service, provides=("map_function"))
        self.rpc_daemon()

    def close(self):
        self.channel.close()

    def _process_result(self, msg):
        self._reply = loads(msg.body)
        #print "received result",self._reply['index'],self._reply['result']
    @daemon
    def _send_map(self, items, mapid):
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

            body = dumps(dict(index=i,value=v,mapid=mapid))
            msg = amqp.Message(body, reply_to=self.reply_queue, delivery_mode=1)
            self.map_channel.basic_publish(msg, exchange=self.exchange,
                                           routing_key=self.map_queue)

    def _send_function(self, function_str, destination):
        msg = amqp.Message(function_str, delivery_mode=1)
        self.map_channel.basic_publish(msg,
                                       exchange=self.exchange,
                                       routing_key=destination)
    def cancel(self):
        """
        Stop a running map.
        """
        raise NotImplementedError()
        # Need to clear the queued items and notify async that no more results.
        # Messages in transit need to be ignored, which probably means tagging
        # each map header with a call number so that previous calls don't
        # get confused with current calls.
        self.reply_channel.basic_publish(msg)

    def async(self, fn, items):
        function_str = dumps(fn)
        current_map = md5sum(function_str)
        items = list(items) # make it indexable
        self.num_items = len(items)
        # Queue items in separate thread so we can start receiving results
        # before all items are even queued
        self.num_processed = 0
        publisher = self._send_map(items, mapid = current_map)
        received = set()
        for i in items:
            while True:
                self.reply_channel.wait()
                mapid = self._repy['mapid']
                if 'sendfunction' in self._reply:
                    destination = self._reply['sendfunction']
                    if mapid == current_map:
                        content = function_str
                    else:
                        content = ""
                    self._send_function(content, mapid, destination)
                elif 'result' in self._reply:
                    idx = self._reply['index']
                    if mapid == current_map:
                        if idx not in received:
                            received.add(idx) # Track responses
                            break
                        else:
                            pass # Ignore duplicates
                    else:
                        pass # ignore late responders
                else:
                    print "ignoring unexpected message"
            result = self._reply['result']
            #print "received %d %g"%(idx,result)
            self.num_processed = i

            ## USE_LOCKS_TO_THROTTLE
            if self.num_queued - self.num_processed < config.MAX_QUEUE - 10:
                # Ten at a time go through for slow processes
                self._throttle.acquire()
                self._throttle.notify()
                self._throttle.release()

            yield idx,result
        publisher.join()
    def __call__(self, fn, items):
        result = list(self.async(fn, items))
        result = list(sorted(result,lambda x,y: cmp(x[0],y[0])))
        return zip(*result)[1]
