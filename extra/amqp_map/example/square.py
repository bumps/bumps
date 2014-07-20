import time

import numpy as np

from amqp_map.config import SERVICE_HOST
from amqp_map.core import connect, Mapper

server = connect(SERVICE_HOST)
square = Mapper(server, "square")

#print square(xrange(5,10))

#for i,v in square.async(xrange(-20,-15)): print i,v

t0 = time.time()
n=10000
print "start direct",n
[ x*x for x in xrange(n)]
print "direct time",1000*(time.time()-t0)/n,"ms/call"


t0 = time.time()
n=100
print "start big",n
square([x*np.ones(3) for x in xrange(n)])
print "remote time",1000*(time.time()-t0)/n,"ms/call"
