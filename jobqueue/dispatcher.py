
from datetime import datetime, timedelta
import logging

from sqlalchemy import and_, or_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from . import runjob, store, db, notify
from .db import Job, ActiveJob

class Scheduler(object):
    def __init__(self):
        db.connect()

    def jobs(self, status=None):
        session = db.Session()
        if status:
            jobs = (session.query(Job)
                .filter(Job.status==status)
                .order_by(Job.priority)
                )
        else:
            jobs = (session.query(Job)
                .order_by(Job.priority)
                )
        return [j.id for j in jobs]
    def submit(self, request, origin):
        session = db.Session()
        # Find number of jobs for the user in the last 30 days
        n = (session.query(Job)
            .filter(or_(Job.notify==request['notify'],Job.origin==origin))
            .filter(Job.date >= datetime.utcnow() - timedelta(30))
            .count()
            )
        #print "N",n
        job = Job(name=request['name'],
                  notify=request['notify'],
                  origin=origin,
                  priority=n)
        session.add(job)
        session.commit()
        store.create(job.id)
        store.put(job.id,'request',request)
        return job.id

    def _getjob(self, id):
        session = db.Session()
        return session.query(Job).filter(Job.id==id).first()

    def results(self, id):
        job = self._getjob(id)
        try:
            return runjob.results(id)
        except KeyError:
            if job:
                return { 'status': job.status }
            else:
                return { 'status': 'UNKNOWN' }

    def status(self, id):
        job = self._getjob(id)
        return job.status if job else 'UNKNOWN'

    def info(self,id):
        request = store.get(id,'request')
        return request

    def cancel(self, id):
        session = db.Session()
        (session.query(Job)
             .filter(Job.id==id)
             .filter(Job.status.in_('ACTIVE','PENDING'))
             .update({ 'status': 'CANCEL' })
             )
        session.commit()

    def delete(self, id):
        """
        Delete any external storage associated with the job id.  Mark the
        job as deleted.
        """
        session = db.Session()
        (session.query(Job)
             .filter(Job.id == id)
             .update({'status': 'DELETE'})
             )
        store.destroy(id)

    def nextjob(self, queue):
        """
        Make the next PENDING job active, where pending jobs are sorted
        by priority.  Priority is assigned on the basis of usage and the
        order of submissions.
        """
        session = db.Session()

        # Define a query which returns the lowest job id of the pending jobs
        # with the minimum priority
        _priority = select([func.min(Job.priority)],
                           Job.status=='PENDING')
        min_id = select([func.min(Job.id)],
                        and_(Job.priority == _priority,
                             Job.status == 'PENDING'))

        for _ in range(10): # Repeat if conflict over next job
            # Get the next job, if there is one
            try:
                job = session.query(Job).filter(Job.id==min_id).one()
                #print job.id, job.name, job.status, job.date, job.start, job.priority
            except NoResultFound:
                return {'request': None}

            # Mark the job as active and record it in the active queue
            (session.query(Job)
             .filter(Job.id == job.id)
             .update({'status': 'ACTIVE',
                      'start': datetime.utcnow(),
                      }))
            activejob = db.ActiveJob(jobid=job.id, queue=queue)
            session.add(activejob)

            # If the job was already taken, roll back and try again.  The
            # first process to record the job in the active list wins, and
            # will change the job status from PENDING to ACTIVE.  Since the
            # job is no longer pending, the  so this
            # should not be an infinite loop.  Hopefully if the process
            # that is doing the transaction gets killed in the middle then
            # the database will be clever enough to roll back, otherwise
            # we will never get out of this loop.
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                continue
            break
        else:
            logging.critical('dispatch could not assign job %s'%job.id)
            raise IOError('dispatch could not assign job %s'%job.id)

        request = store.get(job.id,'request')
        # No reason to include time; email or twitter does that better than
        # we can without client locale information.
        notify.notify(user=job.notify,
                      msg=job.name+" started",
                      level=1)
        return { 'id': job.id, 'request': request }

    def postjob(self, id, results):
        # TODO: redundancy check, confirm queue, check sig, etc.

        # Update db
        session = db.Session()
        (session.query(Job)
            .filter(Job.id == id)
            .update({'status': results.get('status','ERROR'),
                     'stop': datetime.utcnow(),
                     })
            )
        (session.query(ActiveJob)
            .filter(ActiveJob.jobid == id)
            .delete())
        try:
            session.commit()
        except:
            session.rollback()

        # Save results
        store.put(id,'results',results)

        # Post notification
        job = self._getjob(id)
        if job.status == 'COMPLETE':
            if 'value' in results:
                status_msg = " ended with %s"%results['value']
            else:
                status_msg = " complete"
        elif job.status == 'ERROR':
            status_msg = " failed"
        elif job.status == 'CANCEL':
            status_msg = " cancelled"
        else:
            status_msg = " with status "+job.status
        # Note: no reason to include time; twitter or email will give it
        # Plus, doing time correctly requires knowing the locale of the
        # receiving client.
        notify.notify(user=job.notify,
                      msg=job.name+status_msg,
                      level=2)
