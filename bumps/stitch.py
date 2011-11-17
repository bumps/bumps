"""
Data stitching for reflectometry.

Join together datasets yielding unique sorted Q.
"""

from numpy import hstack, vstack, argsort, sum, sqrt

def stitch(probes, same_Q = 0.001, same_dQ = 0.001):
    """
    Stitch together multiple measurements into one.

    *probes* a list of datasets with Q,dQ,R,dR attributes
    *same_Q* minimum point separation (default is 0.001).
    *same_dQ* minimum change in resolution that may be averaged (default is 0.001).

    Wavelength and angle are not preserved since different points with the
    same Q,dQ may have different wavelength/angle inputs, particularly for
    time of flight instruments.

    WARNING: the returned Q values may be data dependent, with two measured
    sets having different Q after stitching, even though the measurement
    conditions are identical!!

    Either add an intensity weight to the datasets::

        probe.I = slitscan

    or use interpolation if you need to align two stitched scans::

        Q1,dQ1,R1,dR1 = stitch([a1,b1,c1,d1])
        Q2,dQ2,R2,dR2 = stitch([a2,b2,c2,d2])
        Q2[0],Q2[-1] = Q1[0],Q1[-1] # Force matching end points
        R2 = numpy.interp(Q1,Q2,R2)
        dR2 = numpy.interp(Q1,Q2,dR2)
        Q2 = Q1

    WARNING: the returned dQ value underestimates the true Q, depending on
    the relative weights of the averaged data points.
    """
    if same_dQ is None: same_dQ = same_Q
    Q = hstack(p.Q for p in probes)
    dQ = hstack(p.dQ for p in probes)
    R = hstack(p.R for p in probes)
    dR = hstack(p.dR for p in probes)
    if all(hasattr(p,'I') for p in probes):
        weight = hstack(p.I for p in probes)
    else:
        weight = R/dR**2  # R/dR**2 is approximately the intensity

    # Sort the data by increasing Q
    idx = argsort(Q)
    data = vstack((Q,dQ,R,dR,weight))
    data = data[:, idx]
    Q = data[0, :]
    dQ = data[1, :]

    # Skip through the data looking for regions of overlap.
    keep = []
    n, last, next = len(Q), 0, 1
    while next < n:
        while next < n and abs(Q[next]-Q[last]) <= same_Q:
            next += 1
        if next - last == 1:
            # Only one point, so no averaging necessary
            keep.append(last)
        else:
            # Pick the Q in [last:next] with the best resolution and average
            # them using Poisson averages.  Repeat until all points are used
            remainder = data[:,last:next]
            avg = []
            while remainder.shape[1] > 0:
                best_dQ = min(remainder[1,:])
                idx = (remainder[1,:]-best_dQ <= same_dQ)
                avg.append(poisson_average(remainder[:,idx]))
                remainder = remainder[:,~idx]
            # Store the result in worst to best resolution order.
            for i,d in enumerate(reversed(avg)):
                data[:,last+i] = d
                keep.append(last+i)
        last = next

    return data[:4,keep]

def poisson_average(QdQRdRw):
    """
    Compute the poisson average of R/dR using a set of data points.

    The returned Q,dQ is the weighted average of the inputs::

        Q = sum(Q*I)/sum(I)
        dQ = sum(dQ*I)/sum(I)

    The returned R,dR use Poisson averaging::

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
    # TODO: need better estimate of dQ, with weighted broadening according
    # to the distance of the Q's from the centers.
    Q,dQ,R,dR,weight = QdQRdRw
    w = sum(weight)
    Q = sum(Q*weight)/w
    dQ = sum(dQ*weight)/w
    R = sum(R*weight)/w
    dR = sqrt(R/w)
    #print "averaging",QdQRdR,Q,dQ,R,dR
    return Q,dQ,R,dR,w
