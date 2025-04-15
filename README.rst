==============================================
Bumps: data fitting and uncertainty estimation
==============================================

Bumps provides data fitting and Bayesian uncertainty modeling for inverse
problems.  It has a variety of optimization algorithms available for locating
the most like value for function parameters given data, and for exploring
the uncertainty around the minimum.

Installation is with the usual python installation command::

    pip install bumps

Once the system is installed, you can verify that it is working with::

    bumps doc/examples/peaks/model.py --chisq

Documentation is available at `readthedocs <http://bumps.readthedocs.org>`_. See
`CHANGES.rst <https://github.com/bumps/bumps/blob/master/CHANGES.rst>`_
for details on recent changes.

If a compiler is available, then significant speedup is possible for DREAM using::

    python -m bumps.dream.build_compiled

(If you have installed from source, you must first check out the random123 library)::

    git clone --branch v1.14.0 https://github.com/DEShawResearch/random123.git bumps/dream/random123
    python -m bumps.dream.build_compiled

For now this requires an install from source rather than pip.

|CI| |RTD| |DOI|

.. |CI| image:: https://github.com/bumps/bumps/actions/workflows/test-publish.yml/badge.svg
   :alt: Build status
   :target: https://github.com/bumps/bumps/actions/workflows/test-publish.yml

.. |DOI| image:: https://zenodo.org/badge/18489/bumps/bumps.svg
   :alt: DOI tag
   :target: https://zenodo.org/badge/latestdoi/18489/bumps/bumps

.. |RTD| image:: https://readthedocs.org/projects/bumps/badge/?version=latest
   :alt: Documentation status
   :target: https://bumps.readthedocs.io/en/latest/?badge=latest
