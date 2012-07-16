Bayesian uncertainty modeling for parametric systems, or peak fitting with error bars.

Bumps has a variety of optimization algorithms available for locating
the most like value for function parameters given data, and for exploring
the uncertainty around the minimum.

Installation is with the usual python installation command:

    python setup.py install

This installs the package for all users of the system.  To isolate
the package it is useful to install virtualenv and virtualenv-wrapper.

This allows you to say:

    mkvirtualenv --system-site-packages bumps
    python setup.py develop

Once the system is installed, you can verify that it is working with: 

    bumps doc/examples/peaks/model.py --chisq


