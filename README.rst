==============================================
Bumps: data fitting and uncertainty estimation
==============================================

Bumps provides data fitting and Bayesian uncertainty modeling for inverse
problems.  It has a variety of optimization algorithms available for locating
the most like value for function parameters given data, and for exploring
the uncertainty around the minimum.

Installation is with the usual python installation command:

    pip install bumps

Once the system is installed, you can verify that it is working with: 

    bumps doc/examples/peaks/model.py --chisq

Documentation is available at `readthedocs <http://bumps.readthedocs.org>`_

.. image:: https://zenodo.org/badge/18489/bumps/bumps.svg
   :target: https://zenodo.org/badge/latestdoi/18489/bumps/bumps

Release notes
=============

v0.7.5.8 2016-04-18
-----------------

* accept model.par output from a different model
* show residuals with curve fit output
* only show correlations for selected variables
* show tics on correlations if small number
* improve handling of uncertainty estimate from curvature
* tweak dream algorithm -- maybe improve the acceptance ratio?
* allow model to set visible variables in output
* improve handling of arbitrary probability density functions
* simplify loading of pymc models
* bug fix: improved handling of ill-conditioned fits
* bug fix: avoid copying mcmc chain during run
* bug fix: more robust handling of --time limit
* bug fix: support newer versions of matplotlib and numpy
* miscellaneous tweaks and fixes

v0.7.5.7 2015-09-21
-------------------

* add entropy calculator (still unreliable for high dimensional problems)
* adjust scaling of likelihood (the green line) to match histogram area
* use --samples to specify the number of samples from the distribution
* mark this and future releases with a DOI at zenodo.org

v0.7.5.6 2015-06-03
-------------------

* tweak uncertainty calculations so they don't fail on bad models

v0.7.5.5 2015-05-07
-------------------

* documentation updates

v0.7.5.4 2014-12-05
-------------------

* use relative rather than absolute noise in dream, which lets us fit target
  values in the order of 1e-6 or less.
* fix covariance population initializer

v0.7.5.3 2014-11-21
-------------------

* use --time to stop after a given number of hours
* Levenberg-Marquardt: fix "must be 1-d or 2-d" bug
* improve curvefit interface

v0.7.5.2 2014-09-26
-------------------

* pull numdifftools dependency into the repository

v0.7.5.1 2014-09-25
-------------------

* improve the load_model interface

v0.7.5 2014-09-10
-----------------

* Pure python release
