#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
'''
Disk And Execution MONitor (Daemon)

daemonize()
  Detach the current process from the calling terminal so it can
  run as a service.
startstop()
  Check the first argument to the program.

     ========  ======
     argument  action
     ========  ======
     start     start the daemon
     stop      stop the daemon
     restart   stop the daemon if running, then start it again
     status    display running status
     run       run but not in daemon mode
     ========  ======

Configurable daemon behaviors:

1. The current working directory set to the "/" directory.
2. The current file creation mode mask set to 0.
3. Close all open files (1024).
4. Redirect standard I/O streams to "/dev/null".

Almost none of this is necessary (or advisable) if your daemon
is being started by inetd. In that case, stdin, stdout and stderr are
all set up for you to refer to the network connection, and the fork()s
and session manipulation should not be done (to avoid confusing inetd).
Only the chdir() and umask() steps remain as useful.

References
==========

UNIX Programming FAQ 1.7 How do I get my program to act like a daemon?
http://www.steve.org.uk/Reference/Unix/faq_2.html

Advanced Programming in the Unix Environment
W. Richard Stevens, 1992, Addison-Wesley, ISBN 0-201-56317-7.

History
=======

* 2001/07/10 by Jürgen Hermann
* 2002/08/28 by Noah Spurrier
* 2003/02/24 by Clark Evans
* 2005/10/03 by Chad J. Schroeder
* 2008/11/05 by Paul Kienzle
* 2009/03/15 by Paul Kienzle (restructured text; fix links)

Based on http://code.activestate.com/recipes/66012/
'''

# TODO: generalize for windows and os x
#
# Need to extend daemon to handle Windows and OS X.
#
# startstop should be renamed control
# control should take additional commands for install/remove, presumably
# by querying the filename of the caller, assuming we can track that.
#
# Should support control from xinetd as well as init.d.  The run command
# may do so already.
#
# Mac OS 10.4 and above uses launchd.  The service will need an info.plist
# to describe the interactions and the script to run.
#
# Windows services can be created and manipulated from python, as shown in::
#    http://essiene.blogspot.com/2005/04/python-windows-services.html
#    http://code.activestate.com/recipes/59872/
#    http://code.activestate.com/recipes/551780/
#
# Need to isolate other system dependencies such as /var/log and /var/run
# so the caller doesn't care where the system normally puts the services.
# Should be able to run the service as a user, so first try putting files
# in the traditional place, and if that doesn't work, put them in ~/.service

import sys, os, time, errno
from signal import SIGTERM

UMASK = 0     # Default to completely private files
WORKDIR = "/" # Default to running in '/'
MAXFD = 1024  # Maximum number of file descriptors

if hasattr(os, "devnull"):
    REDIRECT_TO = os.devnull
else:
    REDIRECT_TO = "/dev/null"

# sys.exit() or os._exit()?
# _exit is like exit(), but it doesn't call any functions registered
# with atexit (and on_exit) or any registered signal handlers.  It also
# closes any open file descriptors.  Using exit() may cause all stdio
# streams to be flushed twice and any temporary files may be unexpectedly
# removed.  It's therefore recommended that child branches of a fork()
# and the parent branch(es) of a daemon use _exit().
exit = os._exit

def _close_all():
    """
    Close all open file descriptors.  This prevents the child from keeping
    open any file descriptors inherited from the parent.
    """

    # There is a variety of methods to accomplish this task.
    #
    # Try the system configuration variable, SC_OPEN_MAX, to obtain the maximum
    # number of open file descriptors to close.  If it doesn't exists, use
    # the default value (configurable).
    #
    # try:
    #    maxfd = os.sysconf("SC_OPEN_MAX")
    # except (AttributeError, ValueError):
    #    maxfd = MAXFD
    #
    # OR
    #
    # if (os.sysconf_names.has_key("SC_OPEN_MAX")):
    #    maxfd = os.sysconf("SC_OPEN_MAX")
    # else:
    #    maxfd = MAXFD
    #
    # OR
    #
    # Use the getrlimit method to retrieve the maximum file descriptor number
    # that can be opened by this process.  If there is not limit on the
    # resource, use the default value.

    import resource              # Resource usage information.
    maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if maxfd == resource.RLIM_INFINITY:
        maxfd = MAXFD

    # Iterate through and close all the file descriptors
    for fd in range(0, maxfd):
        try: os.close(fd)
        except OSError: pass # (ignored) fd wasn't open to begin with


def daemonize(stdout=REDIRECT_TO, stderr=None, stdin=REDIRECT_TO,
              pidfile=None, startmsg = 'started with pid %s' ):
    """
    This forks the current process into a daemon.

    The stdin, stdout, and stderr arguments are file names that will
    be opened and be used to replace the standard file descriptors
    in sys.stdin, sys.stdout, and sys.stderr.
    These arguments are optional and default to /dev/null.

    Note that stderr is opened unbuffered, so if it shares a file with
    stdout then interleaved output may not appear in the order you expect.
    """

    # Fork a child process so the parent can exit.  This returns control to
    # the command-line or shell.  It also guarantees that the child will not
    # be a process group leader, since the child receives a new process ID
    # and inherits the parent's process group ID.  This step is required
    # to insure that the next call to os.setsid is successful.
    try:
        pid = os.fork()
        if pid > 0: exit(0) # Exit first parent.
    except OSError, e:
        raise Exception, "[%d] %s" % (e.errno, e.strerror)

    # Decouple from parent environment.
    os.chdir(WORKDIR)  # Make sure we are not holding a directory open
    os.umask(UMASK)    # Clear the inherited mask
    os.setsid()        # Make this the session leader

    # Fork a second child and exit immediately to prevent zombies.  This
    # causes the second child process to be orphaned, making the init
    # process responsible for its cleanup.  And, since the first child is
    # a session leader without a controlling terminal, it's possible for
    # it to acquire one by opening a terminal in the future (System V-
    # based systems).  This second fork guarantees that the child is no
    # longer a session leader, preventing the daemon from ever acquiring
    # a controlling terminal.
    try:
        pid = os.fork()
        if pid > 0: exit(0) # Exit second parent.
    except OSError, e:
        raise Exception, "[%d] %s" % (e.errno, e.strerror)

    # Save pid
    pid = str(os.getpid())
    if pidfile:  # Make sure pidfile is written cleanly before close_all
        fd = open(pidfile,'w+')
        fd.write("%s\n" % pid)
        fd.flush()
        fd.close()

    # Print start message and flush output
    if startmsg: sys.stderr.write("\n%s\n" % startmsg % pid)
    sys.stdout.flush()
    sys.stderr.flush()

    # Close all but the standard file descriptors.
    #_close_all()  # hmmm...interferes with file output selection

    # Redirect standard file descriptors.
    if not stderr: stderr = stdout
    fin = file(stdin, "r")
    fout = file(stdout, "a+")
    ferr = file(stderr, "a+")
    os.dup2(fin.fileno(), 0)
    os.dup2(fout.fileno(), 1)
    os.dup2(ferr.fileno(), 2)

def readpid(pidfile):
    try:
        pf  = file(pidfile,'r')
        pid = int(pf.read().strip())
        pf.close()
    except IOError:
        pid = None
    return pid

def process_is_running(pid):
    """
    Check if the given process is running.

    Note that this just checks that the pid is in use; since process
    ids can be reused, this isn't a reliable test.
    """
    # Signal the process with 0.  If successful or if fails because
    # you don't have permission, then the process is alive otherwise
    # the process is dead.
    try:
        os.kill(pid, 0)
        return 1
    except OSError, err:
        return err.errno == errno.EPERM

def startstop(stdout=REDIRECT_TO, stderr=None, stdin=REDIRECT_TO,
              pidfile='pid.txt', startmsg = 'started with pid %s' ):
    """
    Process start/stop/restart/status/run commands.

    Start/stop/restart allow the process to be used as an init.d service.
    Run runs the process without daemonizing, e.g., from inittab.
    """
    if len(sys.argv) > 1:
        action = sys.argv[1]
        pid = readpid(pidfile)
        if 'stop' == action or ('restart' == action and pid):
            if not pid:
                msg = "Could not stop, pid file '%s' missing."%pidfile
                sys.stderr.write('%s\n'%msg)
                sys.exit(1)

            try:
                while 1:
                    os.kill(pid,SIGTERM)
                    time.sleep(1)
            except OSError, err:
                err = str(err)
                if err.find("No such process") > 0:
                    os.remove(pidfile)
                    if 'stop' == action:
                        sys.exit(0)
                    # Fall through to the start action
                    pid = None
                else:
                    raise  # Reraise if it is not a "No such process" error

        if action in ['start', 'restart', 'run']:
            if pid:
                msg = "Start aborted since pid file '%s' exists."%pidfile
                sys.stderr.write('%s\n'%msg)
                sys.exit(1)
            # Only return when we are ready to run the main program
            # Otherwise use sys.exit to end.
            if action != 'run':
                daemonize(stdout=stdout,stderr=stderr,stdin=stdin,
                          pidfile=pidfile,startmsg=startmsg)
            return

        if action  == 'status':
            if not pid:
                status='stopped'
            elif not process_is_running(pid):
                status="failed, but pid file '%s' still exists."%pidfile
            else:
                status='running with pid %d'%pid
            sys.stderr.write('Status: %s\n'%status)
            sys.exit(0)

    print "usage: %s start|stop|restart|status|run" % sys.argv[0]
    sys.exit(2)

def test():
    """
    This is an example main function run by the daemon.
    This prints a count and timestamp once per second.
    """
    sys.stdout.write ('Message to stdout...\n')
    sys.stderr.write ('Message to stderr...\n')
    c = 0
    while 1:
        sys.stdout.write ('%d: %s\n' % (c, time.ctime(time.time())) )
        sys.stdout.flush()
        c = c + 1
        time.sleep(1)

if __name__ == "__main__":
    startstop(stdout='/tmp/daemon.log', pidfile='/tmp/daemon.pid')
    test()
