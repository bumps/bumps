"""
Available services.

During configuration of your worker you can add and remove services from
the system using, for example::

    from jobqueue import services

    del services.fitter

    import integration
    services.integrate = integration.service.service
"""

# TODO: modify runjob so that services can be downloaded
# TODO: support over the wire transport for privileged users

def fitter(request):
    from refl1d.fitservice import fitservice
    return fitservice(request)

def count(request):
    print "counting"
    total = 0
    for i in range(request['data']):
        total += 1
    print "done"
    return total
