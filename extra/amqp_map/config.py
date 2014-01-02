"""
Exchange information.


This information should be filled in when connecting to a service.

Some of this should be filled from
Note that the host used for job management and status updates is
going to be different from that used for mapping operations within
the job.

*CLIENT_HOST* | "user:password@host:port/virtual_host"
    Host for messages to and from the job manager and computation monitor.
    The user supplies this when they make a connection.
*SERVICE_HOST* | "user:password@host:port/virtual_host"
    Host for messages within the computation.  The administrator supplies
    this when the configure the compute cluster.
*EXCHANGE* | string
    Exchange name to use for the system.
*MAX_QUEUE* | int
    The maximum number of messages any process should have outstanding.  This
    should be somewhat greater than the number of computers in the cluster,
    but not so large that the computation saturates the exchange.
"""
CLIENT_HOST = "guest:guest@sparkle.ncnr.nist.gov:5672/"
SERVICE_HOST = "guest:guest@sparkle.ncnr.nist.gov:5672/"
EXCHANGE = "park"
MAX_QUEUE = 1000
