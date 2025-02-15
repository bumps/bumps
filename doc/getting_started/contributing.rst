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


Making Changes
==============

Pre-commit hooks
----------------

Once you have installed the code with ``pip install -e .[dev]``, you can make changes to the code.
Note that refl1d uses `pre-commit <https://pre-commit.com/>`_ to run
automated checks and linting/formatting on the code before it is committed.

First, activate the Python environment in which you installed refl1d.
Then, install the pre-commit hooks by running::

    pre-commit install

This will install the pre-commit hooks in your git repository.
The pre-commit hooks will run every time you commit changes to the repository.
If the checks fail, the commit will be aborted.

You can run the checks manually by running::

    pre-commit run

To see what actions are being run, inspect the `.pre-commit-config.yaml` file in the root of the repository.

Simple patches
--------------

If you want to make one or two tiny changes, it is easiest to clone the
repository, make the changes, then send a patch.  This is the simplest way
to contribute to the project.

To run the package from the source tree use the following::

    cd bumps
    python run.py

This will first build the package into the build directory then run it.
Any changes you make in the source directory will automatically be used in
the new version.

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
    pip install pytest pytest-cov
    pytest

When all the tests pass, issue a pull request from your github account.

Please make sure that the documentation is up to date, and can be properly
processed by the sphinx documentation system.  See `_docbuild` for details.

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
