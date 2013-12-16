.. _server-installation:

*******************
Server installation
*******************

.. contents:: :local:

Bumps jobs can be submitted to a remote batch queue for processing.  This
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

If you are setting up a local cluster for performing Bumps analysis then you 
will need to read this section, otherwise you can continue to the next section.

Assuming that the bumps server is installed as user 'bumps' in a virtualenv 
of ~/bumpserve, MPLCONFIGDIR is set to ~/bumpserve/.matplotlib,
and bumpworkd has been configured, you can start with the following profile::

    TODO: fill in some details on bumps server

Job Controller
==============

:mod:`jobqueue` is an independent package within bumps.  It implements
an http API for interacting with jobs.

It is implemented as a WSGI python application using
`Flask <http://flask.pocoo.org>`_

Here is our WSGI setup for apache for our reflectometry modeling service::

    <VirtualHost *:80>
        ServerAdmin pkienzle@nist.gov
        ServerName www.reflectometry.org
        ServerAlias reflectometry.org
        ErrorLog logs/bumps-error_log
        CustomLog logs/bumps-access_log common

        WSGIDaemonProcess bumps_serve user=pkienzle group=refl threads=3
        WSGIScriptAlias /queue /home/pkienzle/bumps/www/jobqueue.wsgi

        <Directory "/home/pkienzle/bumps/www">
                WSGIProcessGroup bumps_serve
                WSGIApplicationGroup %{GLOBAL}
                Order deny,allow
                Allow from all
        </Directory>

        DocumentRoot /var/www/bumps
        <Directory "/var/www/bumps/">
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
adding bumpworkd to your OS initialization scripts.

Security
========

Because the jobqueue can run without authentication we need to be
especially concerned about the security of our system.  Techniques
such as AppArmor or virtual machines with memory mapped file systems
provide a relatively safe environment to support anonymous computing.

To successfully set up AppArmor, there are a few operations you need.

Each protected application needs a profile, usually stored in
/etc/apparmor.d/path.to.application.  With the reflenv virtural
environment in the reflectometry user, the following profile
would be appropriate for the worker daemon::

    -- /etc/apparmor.d/home.bumps.bumpsenv.bin.bumpworkd
    #include <tunables/global>

    /home/bumps/bumpsenv/bin/bumpworkd {
     #include <abstractions/base>
     #include <abstractions/python>

     /bin/dash cx,
     /home/bumps/bumpsenv/bin/python cx,
     /home/bumps/bumpsenv/** r,
     /home/bumps/bumpsenv/**.{so,pyd} mr,
     /home/bumps/.bumpserve/.matplotlib/* rw,
     /home/bumps/.bumpserve/worker/** rw,
    }

This gives read/execute access to python and its C extensions,
and read access to everything else in the bumps virtual environment.

The rw access to .bumpserve is potentially problematic.  Hostile
models can interfere with each other if they are running at the same time.
In particular, they could inject html into the returned data set which can
effectively steal authentication credentials from other users through
cross site scripting attacks, and so would not be appropriate on an 
authenticated service.  Restricting individual models to their own job
directory at .bumpserve/worker/jobid/** would reduce this risk, but this 
author does not know how to do so without elevating bumpworkd privileges to root.

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

Similar profiles could be created for the job server, and indeed, any web
service you have on your machine to reduce the risk that bugs in your code
can be used to compromise your security, but this is less critical since 
your code is not running in general running with arbitrary user defined functions.

