.. _curvefit-example:

****************
Simple functions
****************

.. contents:: :local:

Bumps allows fits with varying levels of complexity.  Simple fits accept
a function $f(x;p)$ and data $x,y,\sigma_y$, where vector $y$ is the value
measured in conditions $x$, and $\sigma_y$ is the $1-\sigma$ uncertainty in
the measurement.  Bumps also provides a simple wrapper for poisson data
taken from counting statistics, with function $f(x;p)$ and data $x,y$.
sim.py is a simulation of data from a poisson process, showing maximum
likelihood, expected value and variance.

The ode2 example shows how to fit a system of coupled differential equations
where multiple values are tracked at each time step.

.. toctree::

    curve.rst
    poisson.rst
    sim.rst
    ode.rst
    ode2.rst
    weighted.rst
