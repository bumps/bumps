.. _server-installation:

*******************
Server installation
*******************

.. contents:: :local:

Refl-1D jobs can be submitted to a remote batch queue for processing.  This
allows users to share large clusters for faster processing of the data.  The
queue consists of several components.

* job controller

   http service layer which allows users to submit jobs and view results

* queue

   cluster management layer which distributes jobs to the working nodes

* worker

   process monitor which runs a job on the working nodes

* mapper

   mechanism for evaluating R(x_i) for different x_i on separate CPUs

If you are setting up a local cluster for performing reflectometry
fits, then you will need to read this section, otherwise you can
continue to the next section.


Assuming that the refl1d server is installed as user reflectometry in
a virtualenv of ~/reflserv, MPLCONFIGDIR is set to ~/reflserve/.matplotlib,
and reflworkd has been configured, you can start with the following profile::


Job Controller
==============

:mod:`jobqueue` is an independent package within refl1d.  It implements
an http API for interacting with jobs.

It is implemented as a WSGI python application using
`Flask <http://flask.pocoo.org>`_

<VirtualHost *:80>
        ServerAdmin pkienzle@nist.gov
        ServerName www.reflectometry.org
        ServerAlias reflectometry.org
        ErrorLog logs/reflectometry-error_log
        CustomLog logs/reflectometry-access_log common

        WSGIDaemonProcess reflserve user=pkienzle group=refl threads=3
        WSGIScriptAlias /queue /home/pkienzle/reflserve/www/jobqueue.wsgi

        <Directory "/home/pkienzle/reflserve/www">
                WSGIProcessGroup reflserve
                WSGIApplicationGroup %{GLOBAL}
                Order deny,allow
                Allow from all
        </Directory>

        DocumentRoot /var/www/reflectometry
        <Directory "/var/www/reflectometry/">
                AllowOverride All
        </Directory>

</VirtualHost>


There is a choice of two different queuing systems to configure.  If your
environment supports a traditional batch queue you can use it to
manage cluster resources.  New jobs are added to the queue, and
when they are complete, they leave their results in the job results
directory.  Currently only slurm is supported, but supporting torque
as well would only require a few changes.

You can also set up a central dispatcher.  In that case, you will have
remote clusters pull jobs from the server when they are available, and post
the results to the job results directory when they are complete. The remote
cluster may be set up with its own queuing system such as slurm, only
taking a few jobs at a time from the dispatcher so that other clusters
can share the load.


Cluster
=======

If you are using the dispatcher queuing system, you will need to set up
a work daemon on your cluster to pull jobs from the queue.  This requires
adding reflworkerd to your OS initialization scripts.

Security
========

Because the jobqueue can run without authentication we need to be
especially concerned about the security of our system.  Techniques
such as AppArmor or virtual machines with memory mapped file systems
provide a relatively safe environment to support anonymous computing.

.. note::

  It easy to add authentication to flask, but we can avoid it if our
  community plays nice --- every beamline should supply sufficient
  compute power to host their user base, either directly or through one of
  the many cloud computing services.

To successfully set up AppArmor, there are a few operations you need.

Each protected application needs a profile, usually stored in
/etc/apparmor.d/path.to.application.  With the reflenv virtural
environment in the reflectometry user, the following profile
would be appropriate for the worker daemon::

    -- /etc/apparmor.d/home.reflectometry.reflenv.bin.reflworkd
    #include <tunables/global>

    /home/reflectometry/reflenv/bin/reflworkd {
     #include <abstractions/base>
     #include <abstractions/python>

     /bin/dash cx,
     /home/reflectometry/reflenv/bin/python cx,
     /home/reflectometry/reflenv/** r,
     /home/reflectometry/reflenv/**.{so,pyd} mr,
     /home/reflectometry/.reflserve/.matplotlib/* rw,
     /home/reflectometry/.reflserve/worker/** rw,
    }

This gives read access/execute access to python and its C extensions,
and read access to everything else in the virtual environment.

The rw access to .reflserve is potentially problematic.  Hostile
models can interfere with each other if they are running at the same time.
In particular, they can inject html into the returned data set which can
effectively steal authentication credentials from other users through
cross site scripting attacks, and so would not be appropriate on a closed
server.  Restricting the model to .reflserve/worker/jobid/** would reduce
this risk, but this author does not know how to do so without elevating
reflworkd privileges to root.


A similar profile could be created for the job server, and indeed, any web
service you have on your machine, but this is less critical since it is not
running user models.

Once the profile is in place, restart the apparmor.d daemon to enable it::

    sudo service apparmor restart

You can debug the profile by running a trace while the program runs
unrestricted.  To start the trace, use::

   sudo genprof /path/to/application

Switch to another window then run::

   /path/to/app

When your application is complete, return to the genprof window
and hit 'S' to scan /var/log/syslog for file and network access.
Follow the prompts to update the profile.  The documentation on
`AppArmor on Ubuntu <https://help.ubuntu.com/community/AppArmor>`_
and
`AppArmor on SUSE <http://doc.opensuse.org/products/opensuse/openSUSE/opensuse-security/cha.apparmor.profiles.html>`_
is very helpful here.

To reload a profile after running the trace, use::

     sudo apparmor_parser -r /etc/apparmor.d/path.to.application

To delete a profile that you no longer need::

     sudo rm /etc/apparmor.d/path.to.application
     sudo service apparmor restart
