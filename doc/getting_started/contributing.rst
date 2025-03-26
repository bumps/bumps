.. _contributing:

********************
Contributing Changes
********************

.. contents:: :local:


The bumps package is a community project, and we welcome contributions from anyone.
The package is developed collaboratively on `Github <https://github.com>`_ - if
you don't have an account yet, you can sign up for free.
For direct write access to the repository, it is required that your account have
`two-factor authentication enabled <https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa>`_.
You may also want to configure your account to use
`SSH keys <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_
for authentication.

The best way to contribute to the bumps package is to work
from a copy of the source tree in the revision control system.

The bumps project is hosted on github at:

    https://github.com/bumps/bumps

You will need the git source control software for your computer.  This can
be downloaded from the `git page <http://www.git-scm.com/>`_, or you can use
an integrated development environment (IDE) such as PyCharm or VS Code, which
may have git built in.


Getting the Code
================

To get the code, you will need to clone the repository.  If you are planning
on making only a few small changes, you can clone the repository directly,
make the changes, document and test, then send a patch (see `Simple patches <#Simple-patches>`_ below).

If you are planning on making larger changes, you should fork the repository
on github, make the changes in your fork, then issue a pull request to the
main repository (see `Larger changes <#Larger-changes>`_ below).

.. note::

    If you are working on a fork, the clone line is slightly different::

        git clone https://github.com/YourGithubAccount/bumps


    You will also need to keep your fork up to date
    with the main repository.  You can do this by adding the main repository
    as a remote, fetching the changes, then merging them into your fork.

    .. code-block:: bash

        # Add the main repository as a remote
        git remote add bumps

        # Fetch the changes from the main repository
        git fetch bumps

        # Merge the changes into your fork
        git merge bumps/master

        # Push the changes to your fork
        git push

Run from source
===============

When working from the source tree it is helpful to install bumps in developer mode.
This allows you to change the files in place and see the changes the next time
you run the program. It also includes all the additional packages needed for
building documentation and testing.

To set up and install the developer version use::

    cd bumps
    conda create --name bumps-dev python
    conda activate bumps-dev
    pip install -e .[dev]

This puts bumps and bumps-webview on your path while leaving the files in place.
Any changes you do to the files will appear when you next run the program.

The webview client uses modern web technologies such as TypeScript, Vue.js, and Plotly.
These are pre-compiled and included with the python wheel for pip installs and also in the
binary installers, but when running from source you will need to build the client package
before starting the server.

To install `nodejs` and build the client use::

    conda install nodejs
    python -m bumps.webview.build_client

This will download the necessary dependencies to build the client package and
save it to the `bumps/webview/client/dist` directory.
You need to run the `build_client` command whenever you change the javascript for the webview interface.

If you already have a python environment with the necessary dependencies and
you don't want to install the package into your environment (for example,
because you are testing out a fork in another source tree), then you can
change to the bumps directory and run the package in place::

    python -m bumps.cli ... # for the command line interface
    python -m bumps.webview.server ... # for the webview interface

.. _docbuild:

Building Documentation
======================

The HTML documentation requires the sphinx, docutils, and jinja2 packages in python.

The PDF documentation requires a working LaTex installation, including the following packages:

    multirow, titlesec, framed, threeparttable, wrapfig,
    collection-fontsrecommended

To build the documentation use::

    (cd doc && make clean html pdf)

Windows users please note that this only works with a unix-like environment
such as *gitbash*, *msys* or *cygwin*.  There is a skeleton *make.bat* in
the directory that will work using the *cmd* console, but it doesn't yet
build PDF files.

You can see the result of the doc build by pointing your browser to::

    bumps/doc/_build/html/index.html
    bumps/doc/_build/latex/Bumps.pdf

ReStructured text format does not have a nice syntax for superscripts and
subscripts.  Units such as |g/cm^3| are entered using macros such as
\|g/cm^3| to hide the details.  The complete list of macros is available in

        doc/sphinx/rst_prolog

In addition to macros for units, we also define cdot, angstrom and degrees
unicode characters here.  The corresponding latex symbols are defined in
doc/sphinx/conf.py.

Making Changes
==============

Simple patches
--------------

If you want to make one or two tiny changes, it is easiest to clone the
repository, make the changes, then send a patch.  This is the simplest way
to contribute to the project.

As you make changes to the package, you can see what you have done using git::

    git status
    git diff

Please update the documentation and add tests for your changes.  We use
doctests on all of our examples so that we know our documentation is correct.
More thorough tests are found in test directory. You can run these tests via pytest,
or via the convenience Makefile target::

    pytest
    # or
    make test

When all the tests run, create a patch and send it to paul.kienzle@nist.gov::

    git diff > patch


Pre-commit hooks
----------------

Bumps uses `pre-commit <https://pre-commit.com/>`_ to run
automated checks and linting/formatting on the code before it is committed.

First, activate the Python environment in which you installed bumps.
Then, install the pre-commit hooks by running::

    pre-commit install

This will install the pre-commit hooks in your git repository.
The pre-commit hooks will run every time you commit changes to the repository.
If the checks fail, the commit will be aborted.

You can run the checks manually by running::

    pre-commit run

To see what actions are being run, inspect the `.pre-commit-config.yaml` file in the root of the repository.


Larger changes
--------------

For a larger set of changes, you should fork bumps on github, and issue pull
requests for each part.

After you have tested your changes, you will need to push them to your github
fork::

    git commit -a -m "short sentence describing what the change is for"
    git push

Good commit messages are a bit of an art.  Ideally you should be able to
read through the commit messages and create a "what's new" summary without
looking at the actual code.

Make sure your fork is up to date before issuing a pull request.  You can
track updates to the original bumps package using::

    git remote add bumps https://github.com/bumps/bumps
    git fetch bumps
    git merge bumps/master
    git push

When making changes, you need to take care that they work on different
versions of python. Using conda makes it convenient to maintain multiple independent
environments. You can create a new environment for testing with, for example::

    conda create -n py312 python=3.12
    conda activate py312
    pip install -e .[dev]
    pytest

When all the tests pass, issue a pull request from your github account.

Please make sure that the documentation is up to date, and can be properly
processed by the sphinx documentation system.  See `_docbuild` for details.


Building an installer (all platforms)
=====================================

To build a packed distribution for Windows, you will need to install
conda-pack in your base conda environment.  If you don't already have
a base interpreter, install that as well (e.g. on Windows) from
conda-forge::

    conda install -c conda-forge conda-pack bash

Then you can build the packed distribution using::

    bash extra/build_conda_packed.sh

This will create a packed distribution in the dist directory.

Creating a New Release
======================

A developer with maintainer status can tag a new release and publish a package to the `Python
Package Index (PyPI) <https://pypi.org/project/bumps/>`_. Bumps uses
`versioningit <https://versioningit.readthedocs.io/>`_ to generate the version number
from the latest tag in the git repository.

1. Update the local copy of the master branch::

    $ # update information from all remotes
    $ git fetch -p -P -t --all
    $ # update local copy of master
    $ git checkout master
    $ git rebase origin/master
    $ # check the current version number (latest tag v0.9.3 + 656 commits)
    $ versioningit
    0.9.4.dev656

2. Add release notes and commit to master.

3. Create the new tag and push it to the remote. Pushing a tag starts the GitHub workflow job to
publish to PyPI (defined in `.github/workflows/test-publish.yml
<https://github.com/bumps/bumps/blob/master/.github/workflows/test-publish.yml>`_)::

    $ git tag v1.0.0
    $ versioningit
    1.0.0
    $ git push origin --tags master
