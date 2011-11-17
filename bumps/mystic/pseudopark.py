print "mystic.pseudopark is untested"
_ = '''
import pickle
class Problem: pass

class MapWorker:
    def config(self, request):
        """
        Prepare the worker for running.  This method does expensive operations
        such as preloading files, etc.  The problem is registered with the
        mapper so that as new workers come into the computation they can
        configure themselves to evaluate a sequence of work units, all of
        which are independent of each other.
        """
        pass

    def reconfig(self, request):
        """
        For steerable jobs, the configuration may change.  The mapper is
        informed of this change, and on subsequent map requests, the
        individual workers are told to update their configuration.
        """

    def evalpop(self, population):
        """
        Evaluate a population array, one value per row, returning a vector
        of results.  The default behaviour is to call eval on each individual
        in the population, but subclasses can use more efficient vector
        implementations if they are available.
        """
        return [self.eval(p) for p in population]

    def eval(self, p):
        """
        Evaluate a single member of the population
        """
        raise NotImplementedError

class Callable(MapWorker):
    def config(self, request):
        self.fn = request
    def reconfig(self, request):
        self.fn = request
    def eval(self, p):
        return self.fn(p)

class VectorCallable(Problem):
    def config(self, request):
        self.fn = request
    def reconfig(self, request):
        self.fn = request
    def evalpop(self, pop):
        return self.fn(pop)



class Mapper:
    def register(self, request):
        """
        Identify a new request, returning the request id
        """
        # id(request) will change each time when the call comes over
        # the wire because deserialization creates a new object.  Therefore
        # we need to do lookups based on the string representation of the
        # object rather than the memory location that id() returns.
        s = pickle.dumps(request)
        sid = hash(s)
        if sid not in self._requests:
            self._requests[sid] = s
        else:
            # We are going to assume that the hash function for serialized
            # requests has an extremely low probability of collision.
            # Unfortunately the consequences of a collision are nasty (the
            # wrong problem will be optimized for example).  To protect
            # against this, lets make sure that our hash matches our
            # request.
            if self._requests[sid] != s:
                raise CollisionError("two problems happen to match")
        return sid
    def unregister(self, sid):
        #TODO: ??? called by whom? When?
        pass
    def submit(self, sid, v, callback, tag=None):
        if sid not in self._running:
            self.startworkers(self_requests[sid])
            self._running.append(sid)
        #results = CollectWork(callback=callback,tag=tag, ...)
        self.queue.put(sid,v)
        return
    def onworkresult(self, fv):
        result.values.append(fv)
        if result.complete:
            result.callback(result.values, tag=result.tag)

# Client side
class MapResult:
    def __init__(self, queue):
        self.queue = queue
    def wait(self):
        return self.queue.get()



class Handler:
    def __init__(self):
        """
        mapper is a possibly remote object which follows the Mapper
        interface
        """
        self._mapper = lookup_mapper_or_send_it_as_a_parameter
        self._mapid = {}
    def map(self, f, v):
        # Called in the service thread
        if f not in self._mapids:
            # If calling with a function rather than a map worker,
            # wrap the function in the Callable map worker.
            request = f if isinstance(f,MapWorker) else Callable(f)
            # Register the request with the mapper
            self._mapids[f] = self._mapper.register(request)
        mapid = self._mapids[f]
        result = Queue()
        self._waiting[id(result)] = result
        self._mapper.submit(mapid,v,pyro_id_of_handler.onmapcomplete,
                            tag=id(result))
        return result
    def map_clear(self, f):
        self._mapper.unregister(f)
    def onmapcomplete(self, result,tag=None):
        # Called in the handler thread
        queue = self._waiting[tag]
        queue.put(result)
        del self._waiting[tag]
'''
