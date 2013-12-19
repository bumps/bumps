from __future__ import print_function

from jobqueue.client import connect

DEBUG = True

#server = connect('http://reflectometry.org/queue')
server = connect('http://localhost:5000')

def checkqueue(pending=[], active=[], complete=[]):
    qpending = server.jobs('PENDING')
    qactive = server.jobs('ACTIVE')
    qcomplete = server.jobs('COMPLETE')
    if DEBUG: print("pending",qpending,"active",qactive,"complete",qcomplete)
    #assert pending == qpending
    #assert active == qactive
    #assert complete == qcomplete

long = {'service':'count','data':1000000,
        'name':'long count','notify':'me'}
short = {'service':'count','data':200,
        'name':'short count','notify':'me'}
fail1 = {'service':'count','data':'string',
         'name':'short count','notify':'me'}
fail2 = {'service':'noservice','data':'string',
         'name':'short count','notify':'me'}

job = server.submit(int)
print("submit",job)
#import sys; sys.exit()
checkqueue()
job2 = server.submit(short)
print("submit",job2)
result = server.wait(job['id'], pollrate=10, timeout=120)
print("result",result)
checkqueue()
print("fetch",server.info(job['id']))
print("delete",server.delete(job['id']))
checkqueue()
job3 = server.submit(fail1)
job4 = server.submit(fail2)
print("===incorrect service options")
result = server.wait(job3['id'], pollrate=1, timeout=120)
print(result['error'])
print(result['trace'])
print("===incorrect service")
result = server.wait(job4['id'], pollrate=1, timeout=120)
print(result['error'])
print(result['trace'])
