==============================================
Bumps: data fitting and uncertainty estimation
==============================================

Bumps provides data fitting and Bayesian uncertainty modeling for inverse
problems.  It has a variety of optimization algorithms available for locating
the most like value for function parameters given data, and for exploring
the uncertainty around the minimum.

Installation is with the usual python installation command:

.. code-block:: bash

    pip install bumps

Once the system is installed, you can verify that it is working with:

.. code-block:: bash

    bumps doc/examples/peaks/model.py --chisq
    bumps -h

To start the webview interface use:

.. code-block:: bash

    bumps

Documentation is available at `readthedocs <http://bumps.readthedocs.org>`_. See
`CHANGES.rst <https://github.com/bumps/bumps/blob/master/CHANGES.rst>`_
for details on recent changes.

If a compiler is available, then significant speedup is possible for DREAM using::

.. code-block:: bash

    python -m bumps.dream.build_compiled

If you have installed from source, you must first check out the random123 library::

    git clone --branch v1.14.0 https://github.com/DEShawResearch/random123.git bumps/dream/random123
    python -m bumps.dream.build_compiled

If you have installed from source you will need to build the webview client:

.. code-block:: bash

    pip install nodeenv  # if nodejs is unavailable
    nodeenv --prebuilt -p  # then install it with nodeenv
    python -m bumps.webview.build_client

|CI| |RTD| |DOI|

.. |CI| image:: https://github.com/bumps/bumps/actions/workflows/test-publish.yml/badge.svg
   :alt: Build status
   :target: https://github.com/bumps/bumps/actions/workflows/test-publish.yml

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.594099.svg
   :alt: DOI tag
   :target: https://doi.org/10.5281/zenodo.594099

.. |RTD| image:: https://readthedocs.org/projects/bumps/badge/?version=latest
   :alt: Documentation status
   :target: https://bumps.readthedocs.io/en/latest/?badge=latest
