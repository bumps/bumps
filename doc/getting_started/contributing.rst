.. _contributing:

********************
Contributing Changes
********************

.. contents:: :local:

The best way to contribute to the Bumps package is to work from a copy of
the source tree in the revision control system.

The bumps project is hosted on github at:

    http://github.com/bumps

You can obtain a copy via git using::

    git clone https://github.com/bumps/bumps.git
    cd bumps
    python setup.py develop

By using the *develop* keyword on setup.py, changes to the files in the
package are immediately available without the need to run setup each time
you change the code.

Track updates to the original package using::

    git pull

If you find you need to modify the package, please update the documentation
and add tests for your changes.  We use doctests on all of our examples to
help keep the documentation synchronized with the code.  More thorough tests
are found in the test directory.  Using the the nose test package, you can
run both sets of tests::

    pip install nose
    python2.5 tests.py
    python2.6 tests.py

When all the tests run, generate a patch and send it to pkienzle@nist.gov::

    git diff > patch

Windows user can use `TortoiseGit <http://code.google.com/p/tortoisegit/>`_
package which provides similar operations.

Instead of sending patches, you can set up a github account and create
your own bumps fork.  This allows you to develop code at your leisure with
the safety of source control, and issue pull requests when your code is ready
to merge with the main repository.

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
