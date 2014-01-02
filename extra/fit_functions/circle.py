#!/usr/bin/env python

"""
Minimal disk

References::
    None
"""

from numpy import array, pi, inf, vstack, linspace
from numpy import random, sin, cos, sqrt

random.seed(123)


def disk_coverage(data, cx, cy, r,
                  area_penalty=1, visibility_penalty=1, distance_penalty=1):
    """
    cost function for minimum enclosing circle for a 2D set of points

    There are three penalty terms:

    - *area_penalty* is the cost per unit area of the disk
    - *visibility_penalty* is the cost per point not covered by the disk
    - *distance_penalty* is the weight on the sum squared costs of
      each point to the disk
    """
    if r<0: return inf
    x,y = data
    d = sqrt((x-cx)**2 + (y-cy)**2)
    return area_penalty*pi*r*2 + distance_penalty*sum((d[d>r]-r)**2) + visibility_penalty*sum(d>r)


def outline(N=200, cx=0, cy=0, r=1):
    """
    generate the outline of a circle using N steps.
    """
    theta = linspace(0, 2*pi, N)
    return vstack( (r*cos(theta)+cx, r*sin(theta)+cy) )

def simulate_disk(N, cx=0, cy=0, r=1):
    """
    Generate N random points in a disk
    """
    data = array(list(_disk_generator(N)))
    return vstack( (r*data[:,0]+cx, r*data[:,1]+cy) )

def simulate_circle(N, cx=0, cy=0, r=1):
    """
    generate N random points on a circle
    """

    theta = random.uniform(0,2*pi,size=N)
    return vstack((r*cos(theta)+cx,r*sin(theta)+cy))

def _disk_generator(N):
    for _ in range(N):
        while True:
            x = random.random()*2.-1.
            y = random.random()*2.-1.
            if x*x + y*y <= 1:
                break
        yield x,y


# End of file
