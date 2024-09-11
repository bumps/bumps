.. _developer-release:

######################
Creating a New Release
######################

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

2. Create the new tag and push it to the remote. Pushing a tag starts the GitHub workflow to publish
to PyPI (defined in `.github/workflows/pypi-publish.yml
<https://github.com/bumps/bumps/tree/master/.github/workflows/pypi-publish.yml>`_)::

    $ git tag v1.0.0
    $ versioningit
    1.0.0
    $ git push origin --tags master
