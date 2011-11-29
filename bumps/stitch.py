"""
Data stitching.

Join together datasets yielding unique sorted x.
"""

from numpy import hstack, vstack, argsort, sum, sqrt

def stitch(data, same_x = 0.001, same_dx = 0.001):
    """
    Stitch together multiple measurements into one.

    *data* a list of datasets with x,dx,y,dy attributes
    *same_x* minimum point separation (default is 0.001).
    *same_dx* minimum change in resolution that may be averaged (default is 0.001).

    Wavelength and angle are not preserved since different points with the
    same x,dx may have different wavelength/angle inputs, particularly for
    time of flight instruments.

    WARNING: the returned x values may be data dependent, with two measured
    sets having different x after stitching, even though the measurement
    conditions are identical!!

    Either add an intensity weight to the datasets::

        probe.I = slitscan

    or use interpolation if you need to align two stitched scans::

        x1,dx1,y1,dy1 = stitch([a1,b1,c1,d1])
        x2,dx2,y2,dy2 = stitch([a2,b2,c2,d2])
        x2[0],x2[-1] = x1[0],x1[-1] # Force matching end points
        y2 = numpy.interp(x1,x2,y2)
        dy2 = numpy.interp(x1,x2,dy2)
        x2 = x1

    WARNING: the returned dx value underestimates the true x, depending on
    the relative weights of the averaged data points.
    """
    if same_dx is None: same_dx = same_x
    x = hstack(p.x for p in data)
    dx = hstack(p.dx for p in data)
    y = hstack(p.y for p in data)
    dy = hstack(p.dy for p in data)
    if all(hasattr(p,'I') for p in data):
        weight = hstack(p.I for p in data)
    else:
        weight = y/dy**2  # y/dy**2 is approximately the intensity

    # Sort the data by increasing x
    idx = argsort(x)
    data = vstack((x,dx,y,dy,weight))
    data = data[:, idx]
    x = data[0, :]
    dx = data[1, :]

    # Skip through the data looking for regions of overlap.
    keep = []
    n, last, next = len(x), 0, 1
    while next < n:
        while next < n and abs(x[next]-x[last]) <= same_x:
            next += 1
        if next - last == 1:
            # Only one point, so no averaging necessary
            keep.append(last)
        else:
            # Pick the x in [last:next] with the best resolution and average
            # them using Poisson averages.  Repeat until all points are used
            remainder = data[:,last:next]
            avg = []
            while remainder.shape[1] > 0:
                best_dx = min(remainder[1,:])
                idx = (remainder[1,:]-best_dx <= same_dx)
                avg.append(poisson_average(remainder[:,idx]))
                remainder = remainder[:,~idx]
            # Store the result in worst to best resolution order.
            for i,d in enumerate(reversed(avg)):
                data[:,last+i] = d
                keep.append(last+i)
        last = next

    return data[:4,keep]

def poisson_average(xdxydyw):
    """
    Compute the poisson average of y/dy using a set of data points.

    The returned x,dx is the weighted average of the inputs::

        x = sum(x*I)/sum(I)
        dx = sum(dx*I)/sum(I)

    The returned y,dy use Poisson averaging::

        w = sum(y/dy^2)
        y = sum((y/dy)^2)/w
        dy = sqrt(y/w)

    The above formula gives the expected result for combining two
    measurements, assuming there is no uncertainty in the monitor.

        measure N counts during M monitors
          rate:                   r = N/M
          rate uncertainty:       dr = sqrt(N)/M
          weighted rate:          r/dr^2 = (N/M) / (N/M^2) =  M
          weighted rate squared:  r^2/dr^2 = (N^2/M^2) / (N/M^2) = N

        for two measurements Na, Nb
          w = ra/dra^2 + rb/drb^2 = Ma + Mb
          y = ((ra/dra)^2 + (rb/drb)^2)/w = (Na + Nb)/(Ma + Mb)
          dy = sqrt(y/w) = sqrt( (Na + Nb)/ w^2 ) = sqrt(Na+Nb)/(Ma + Mb)
    """
    # TODO: need better estimate of dx, with weighted broadening according
    # to the distance of the x's from the centers.
    x,dx,y,dy,weight = xdxydyw
    w = sum(weight)
    x = sum(x*weight)/w
    dx = sum(dx*weight)/w
    y = sum(y*weight)/w
    dy = sqrt(y/w)
    #print "averaging",xdxydy,x,dx,y,dy
    return x,dx,y,dy,w
