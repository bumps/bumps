"""
Note: slurm configuration for debian::

    sudo apt-get install slurm-llnl
    sudo vi /etc/slurm-llnl/slurm.conf
    sudo mkdir /var/run/slurm-llnl
    sudo chmod slurm:slurm /var/run/slurm-llnl

Now put it in rc.d::

    sudo update-rc.d munge defaults
    sudo update-rc.d slurm-llnl defaults

Or run the following each time::

    sudo service munge start
    sudo service slurm-llnl start

"""

import os
import subprocess
import multiprocessing

from .jobid import get_jobid
from . import store, runjob

#from park import config
#from park import environment

# Queue status words
_ACTIVE = ["RUNNING", "COMPLETING"]
_INACTIVE = ["PENDING", "SUSPENDED"]
_ERROR = ["CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL"]
_COMPLETE = ["COMPLETED"]


class Scheduler(object):
    def jobs(self, status=None):
        """
        Return a list of jobs on the queue.
        """
        #TODO: list completed but not deactivated as well as completed
        #print "queue"
        output,_ = _slurm_exec('squeue','-o', '%i %M %j')
        #print "output",output
        return output

    def deactivate(self, jobid):
        #TODO: remove the job from the active list
        pass

    # Job specific commands
    def submit(self, request, origin):
        """
        Put a command on batch queue, returning its job id.
        """
        #print "submitting job",jobid
        jobid = get_jobid()
        store.create(jobid)
        store.put(jobid,'request',request)

        service = runjob.build_command(jobid,request)

        num_workers = multiprocessing.cpu_count()
        jobdir = store.path(jobid)
        script = os.path.join(jobdir,"J%s.sh"%jobid)
        #commands = ['export %s="%s"'%(k,v) for k,v in config.env().items()]
        commands = ["srun -n 1 -K -o slurm-out.txt nice -n 19 %s &"%service]
#                     "srun -n %d -K -o kernel.out nice -n 19 %s"%(num_workers,kernel)]
        create_batchfile(script,commands)

        _out,err = _slurm_exec('sbatch',
                            '-n',str(num_workers), # Number of tasks
                            #'-K', # Kill if any process returns error
                            #'-o', 'job%j.out',  # output file
                            '-D',jobdir,  # Start directory
                            script)
        if not err.startswith('sbatch: Submitted batch job '):
            raise RuntimeError(err)
        slurmid = err[28:].strip()
        store.put(jobid,'slurmid',slurmid)
        return jobid

    def results(self, id):
        try:
            return runjob.results(id)
        except KeyError:
            pass

        return { 'status': self.status(id) }

    def info(self,id):
        request = store.get(id,'request')
        request['id'] = id
        return request

    def status(self, id):
        """
        Returns the follow states:
        PENDING   --  Job is waiting to be processed
        ACTIVE    --  Job is busy being processed through the queue
        COMPLETE  --  Job has completed successfully
        ERROR     --  Job has either been canceled by the user or an
                      error has been raised
        """

        # Simple checks first
        jobdir = store.path(id)
        if not os.path.exists(jobdir):
            return "UNKNOWN"
        elif store.contains(id, 'results'):
            return "COMPLETE"

        # Translate job id to slurm id
        slurmid = store.get(id,'slurmid')

        # Check slurm queue for job id
        out,_ = _slurm_exec('squeue', '-h', '--format=%i %T')
        out = out.strip()

        state = ''
        inqueue = False
        if out != "":
            for line in out.split('\n'):
                line = line.split()
                if slurmid == line[0]:
                    state = line[1]
                    inqueue = True
                    break

        if inqueue:
            if state in _ACTIVE:
                return "ACTIVE"
            elif state in _INACTIVE:
                return "PENDING"
            elif state in _COMPLETE:
                return "COMPLETE"
            elif state in _ERROR:
                return "ERROR"
            else:
                raise RuntimeError("unexpected state from squeue: %s"%state)
        else:
            return "ERROR"

    def cancel(self, jobid):
        #print "canceling",jobid
        slurmid = store.get(jobid,'slurmid')
        _slurm_exec('scancel',slurmid)

    def delete(self, jobid):
        #jobdir = store.path(jobid)
        self.cancel(jobid)
        store.destroy(jobid)

    def nextjob(self, queue):
        raise NotImplementedError("SLURM queues do not support work sharing")

    def postjob(self, id, results):
        raise NotImplementedError("SLURM queues do not support work sharing")

def create_batchfile(script, commands):
    """
    Create the batchfile to run the job.
    """
    fid = open(script,'w')
    fid.write("#!/bin/sh\n")
    fid.write("\n".join(commands))
    fid.write("\nwait\n")
    fid.close()
    return script

def _slurm_exec(cmd, *args):
    """
    Run a slurm command, capturing any errors.
    """
    #print "cmd",cmd,"args",args
    process = subprocess.Popen([cmd]+list(args),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out,err = process.communicate()
    if err.startswith(cmd+': error: '):
        raise RuntimeError(cmd+': '+err[15:].strip())
    return out,err
