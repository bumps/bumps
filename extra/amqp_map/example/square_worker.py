
from amqp_map.config import SERVICE_HOST
from amqp_map.core import connect, start_worker

#BUSY = 30000000
#BUSY = 1000000
BUSY = 0
def square(x):
    print "recv'd",x
    #x = float(x)
    for i in xrange(BUSY):
        x = x+i
    for i in xrange(BUSY):
        x = x-i
    return x*x
server = connect(SERVICE_HOST)
start_worker(server, "square", square)
