# Dead code
print "mystic.meshgrid is untested"

#from park Request, Service
class Service: pass
class Request: pass
import numpy

class MeshgridService(Service):
    def prepare(self, problem, dims, steps):
        self.problem = problem
        self.dims = dims
        self.steps = steps
        self.mesh = []

    def run(self, handler):
        mapper = handler.mapper(self.problem)
        x,y = self.steps
        pop = numpy.tile([p.value for p in self.problem.parameters],
                         (len(self.x),1))

        xdim,ydim = self.dims
        pop[:,xdim] = x
        start = len(self.mesh)
        for yi in y[start:]:
            pop[:,ydim] = yi
            self.mesh.append(mapper(pop))
            handler.ready()

        return x,y,numpy.array(self.mesh)

    def checkpoint(self):
        return dict(problem=self.problem,
                    dims=self.dims,
                    steps=self.steps,
                    mesh = self.mesh)

    def restore(self, state):
        self.problem = state['problem']
        self.dims = state['dims']
        self.steps = state['steps']
        self.mesh = state['mesh']
        return

    def progress(self):
        return len(self.mesh), len(self.steps[1]), "rows"

    def cleanup(self):
        pass


# === Client interface ===
def meshgrid(f, p, dims, steps):
    r = Request('MeshGridService',0.0,problem=(f,p),dims=dims,steps=steps)
    return r
