import os
import json
import shutil
from tempfile import NamedTemporaryFile

ROOT = '/var/lib/jobqueue/server/%s'

def tempfile():
    create('temp')
    return NamedTemporaryFile(delete=False, dir=path('temp'))

def path(id):
    return ROOT%id

def create(id):
    #print "making %s"%path(id)
    if not os.path.exists(path(id)):
        os.makedirs(path(id))

def destroy(id):
    shutil.rmtree(path(id))

def put(id, key, value):
    value = json.dumps(value)
    datapath = path(id)
    datafile = os.path.join(datapath,"K-%s.json"%(key))
    try:
        open(datafile,'wb').write(value)
    except:
        raise KeyError("Could not store key %s-%s in %s"%(id,key,datafile))

def get(id, key):
    datapath = path(id)
    datafile = os.path.join(datapath,"K-%s.json"%(key))
    try:
        value = open(datafile,'rb').read()
    except:
        raise KeyError("Could not retrieve key %s-%s"%(id,key))
    #if value == "": print "key %s-%s is empty"%(id,key)
    return json.loads(value) if value != "" else None

def contains(id, key):
    datapath = path(id)
    datafile = os.path.join(datapath,"K-%s.json"%(key))
    return os.path.exists(datafile)

def delete(id, key):
    datapath = path(id)
    datafile = os.path.join(datapath,"K-%s.json"%(key))
    try:
        os.unlink(datafile)
    except:
        raise KeyError("Could not delete key %s-%s"%(id,key))
