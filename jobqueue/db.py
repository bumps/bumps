import os.path

from datetime import datetime

import sqlalchemy as db
from sqlalchemy import Column, ForeignKey, Sequence
from sqlalchemy import String, Integer, DateTime, Float, Enum
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DB_URI = 'sqlite:///'+os.path.expanduser('~/.jobqueue.db')
DEBUG = False

Record = declarative_base()
Session = sessionmaker(autocommit=False)
def connect():
    engine = db.create_engine(DB_URI, echo=DEBUG)
    Record.metadata.create_all(engine)
    Session.configure(bind=engine)

# Job status enum
STATUS = ['PENDING','ACTIVE','CANCEL','COMPLETE','ERROR','DELETE']

class Job(Record):
    """
    *id* : Integer
        Unique id for the job
    *name* : String(80)
        Job name as specified by the user.  This need not be unique.
    *origin* : String(45)
        IP address originating the request
    *date* : DateTime utc
        Request submission time
    *start* : DateTime utc
        Time the request was processed
    *stop* : DateTime utc
        Time the request was completed
    *priority* : Float
        Priority level for the job
    *notify* : String(254)
        Email/twitter notification address
    *status* : PENDING|ACTIVE|CANCEL|COMPLETE|ERROR
        Job status

    The job request, result and any supplementary information are
    stored in the directory indicated by jobid.
    """

    __tablename__ = 'jobs'

    id = Column(Integer, Sequence('jobid_seq'), primary_key=True)
    name = Column(String(80))
    origin = Column(String(45)) # <netinet/in.h> #define INET6_ADDRSTRLEN 46
    date = Column(DateTime, default=datetime.utcnow, index=True)
    start = Column(DateTime)
    stop = Column(DateTime)
    priority = Column(Float, index=True)
    notify = Column(String(254)) # RFC 3696 errata 1690: max email=254
    status = Column(Enum(*STATUS, name="status_enum"), index=True)

    def __init__(self, name, origin, notify, priority):
        self.status = 'PENDING'
        self.name = name
        self.origin = origin
        self.notify = notify
        self.priority = priority

    def __repr__(self):
        return "<Job('%s')>" % (self.name)

class ActiveJob(Record):
    """
    *id* : Integer
        Unique id for the job
    *jobid* : job.id
        Active job
    *queue* : String(256)
        Queue on which the job is running
    *date* : DateTime utc
        Date the job was queued
    """
    # TODO: split queue into its own table, and create an additional table
    # TODO: to track how much work is done by each queue
    __tablename__ = "active_jobs"
    id = Column(Integer, Sequence('activeid_seq'), primary_key=True)
    jobid = Column(Integer, ForeignKey(Job.id), unique=True)
    queue = Column(String(256))
    date = Column(DateTime, default=datetime.utcnow)

    job = relationship(Job, uselist=False)

    def __init__(self, jobid, queue):
        self.jobid = jobid
        self.queue = queue

    def __repr__(self):
        return "<ActiveJob('%s','%s')>" % (self.job_id, self.queue)

class RemoteQueue(Record):
    """
    *id* : Integer
        Unique id for the remote server
    *name* : String(80)
        Name of the remote server
    *public_key* : String(80)
        Public key for the remote server
    """
    __tablename__ = "remote_queues"
    id = Column(Integer, Sequence('remotequeueid_seq'),
                   primary_key=True)
    name = Column(String(80))
    public_key = Column(String(80))
