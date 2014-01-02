import os

#from park import config
JOBID_PATH = os.path.expanduser('~/.jobqueue.job')

FID = None
CURRENT = None
def get_jobid():
    global FID, CURRENT
    if FID is None:
        path = JOBID_PATH
        if not os.path.exists(path):
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            FID = open(path,'w+')
            CURRENT = 0
        else:
            FID = open(path,'r+')
            CURRENT = int(FID.read())
    CURRENT += 1
    FID.seek(0)
    FID.write(str(CURRENT))
    FID.flush()
    return str(CURRENT)
