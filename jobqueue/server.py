# TODO: Add /jobs/<id>/data.zip to fetch all files at once in a zip file format
# TODO: Store completed work in /path/to/store/<id>.zip

import os, sys
import logging
import json
import cPickle as pickle
import flask
from flask import redirect, url_for, flash
from flask import send_from_directory
from werkzeug import secure_filename

from jobqueue import store

app = flask.Flask(__name__)

# ==== File upload specialization ===
# By uploading files into a temporary file provided by store, we
# can then move the files directly into place on the store rather
# than copy them.  This gives us reduced storage, reduced memory
# and reduced cpu.
class Request(flask.Request):
    # Upload directly into temporary files.
    def _get_file_stream(self, total_content_length, content_type,
                         filename=None, content_length=None):
        #print "returning named temporary file for",filename
        return store.tempfile()
app.request_class = Request


# ==== Format download specialialization ===
def _format_response(response, format='json', template=None):
    """
    Return response as a particular format.
    """
    #print "response",response
    if format == 'html':
        if template is None: flask.abort(400)
        return flask.render_template(template, **response)
    elif format == 'json':
        return flask.jsonify(**dict((str(k),v) for k,v in response.items()))
    elif format == 'pickle':
        return pickle.dumps(response)
    else:
        flask.abort(400) # Bad request


@app.route('/jobs.<format>', methods=['GET'])
def list_jobs(format='json'):
    """
    GET /jobs.<format>

    Return a list of all job ids.
    """
    response = dict(jobs=SCHEDULER.jobs())
    return _format_response(response, format, template='list_jobs.html')

@app.route('/jobs/<any(u"pending",u"active",u"error",u"complete"):status>.<format>',
           methods=['GET'])
def filter_jobs(status, format='json'):
    """
    GET /jobs/<pending|active|error|complete>.<format>

    Return all jobs with a particular status.
    """
    response = dict(jobs=SCHEDULER.jobs(status=str(status).upper()))
    return _format_response(response, format, template='list_jobs.html')

@app.route('/jobs.<format>', methods=['POST'])
def create_job(format='json'):
    """
    POST /jobs.<format>

    Schedule a new job, return the job record.

    The POST data should contain::

        {
        notify: "<user@email or @twitterid>",
        service: "<name of service>",
        version: "<service version>",
        name: "<request name>",
        data: "<service data>",
        ...
        }

    The response contains::

        {
        id: <job id>,
        job: <job details>
        }

    Job details is simply a copy of the original request.

    """
    request = flask.request.json
    if request is None: flask.abort(415) # Unsupported media
    id = SCHEDULER.submit(request, origin=flask.request.remote_addr)
    flash('Job %s scheduled' % id)
    response = {'id': id, 'job': SCHEDULER.info(id)}
    #return redirect(url_for('show_job', id=id, format=format))
    return _format_response(response, format=format, template='show_job.html')

@app.route('/jobs/<int:id>.<format>', methods=['GET'])
def show_job(id, format='json'):
    """
    GET /jobs/<id>.<format>

    Get job record by id.

    The response contains::

        {
        id: <job id>,
        job: <job details>
        }

    Job details is simply a copy of the original request.
    """
    response = {'id': id, 'job': SCHEDULER.info(id)}
    return _format_response(response, format=format, template='show_job.html')

@app.route('/jobs/<int:id>/results.<format>', methods=['GET'])
def get_results(id, format='json'):
    """
    GET /jobs/<id>/results.<format>

    Get job results by id.

    Returns::

        {
        id: <job id>
        status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN',
        result: <job value>     (absent if status != COMPLETE)
        trace: <error trace>    (absent if status != ERROR)
        }
    """
    response = SCHEDULER.results(id)
    response['id'] = id
    #print "returning response",response
    return _format_response(response, format=format)

@app.route('/jobs/<int:id>/status.<format>', methods=['GET'])
def get_status(id, format='json'):
    """
    GET /jobs/<id>/status.<format>

    Get job status by id.

    Returns::

        {
        id: <job id>,
        status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN'
        }
    """
    response = { 'status': SCHEDULER.status(id) }
    response['id'] = id
    return _format_response(response, format=format)


@app.route('/jobs/<int:id>.<format>', methods=['DELETE'])
def delete_job(id, format='json'):
    """
    DELETE /jobs/<id>.<format>

    Deletes a job, returning the list of remaining jobs as <format>
    """
    SCHEDULER.delete(id)
    flash('Job %s deleted' % id)
    response = dict(jobs=SCHEDULER.jobs())
    return _format_response(response, format=format, template="list_jobs.html")
    #return redirect(url_for('list_jobs', id=id, format=format))

@app.route('/jobs/nextjob.<format>', methods=['POST'])
def fetch_work(format='json'):
    # TODO: verify signature
    request = flask.request.json
    if request is None: flask.abort(415) # Unsupported media
    job = SCHEDULER.nextjob(queue=request['queue'])
    return _format_response(job, format=format)

@app.route('/jobs/<int:id>/postjob', methods=['POST'])
def return_work(id):
    # TODO: verify signature corresponds to flask.request.form['queue']
    # TODO: verify that work not already returned by another client
    try:
        #print "decoding <%s>"%flask.request.form['results']
        results = json.loads(flask.request.form['results'])
    except:
        import traceback;
        logging.error(traceback.format_exc())
        results = {
            'status': 'ERROR',
            'error': 'No results returned from the server',
            'trace': flask.request.form['results'],
        }
    _transfer_files(id)
    SCHEDULER.postjob(id, results)
    # Should be signalling code 204: No content
    return _format_response({},format="json")

@app.route('/jobs/<int:id>/data/index.<format>')
def listfiles(id, format):
    try:
        path = store.path(id)
        files = sorted(os.listdir(path))
        finfo = [(f,os.path.getsize(os.path.join(path,f)))
                 for f in files if os.path.isfile(os.path.join(path,f))]
    except:
        finfo = []
    response = { 'files': finfo }
    response['id'] = id
    return _format_response(response, format=format, template="index.html")

# TODO: don't allow putfiles without authentication
#@app.route('/jobs/<int:id>/data/', methods=['GET','PUT'])
def putfiles(id):
    if flask.request.method=='PUT':
        # TODO: verify signature
        _transfer_files(id)
    return redirect(url_for('getfile',id=id,filename='index.html'))

@app.route('/jobs/<int:id>/data/<filename>')
def getfile(id, filename):
    as_attachment = filename.endswith('.htm') or filename.endswith('.html')
    if filename.endswith('.json'):
        mimetype = "application/json"
    else:
        mimetype = None

    return send_from_directory(store.path(id), filename,
                               mimetype=mimetype, as_attachment=as_attachment)

#@app.route('/jobs/<int:id>.<format>', methods=['PUT'])
#def update_job(id, format='.json'):
#    """
#    PUT /job/<id>.<format>
#
#    Updates a job using data from the job submission form.
#    """
#    book = Book(id=id, name=u"I don't know") # Your query
#    book.name = request.form['name'] # Save it
#    flash('Book %s updated!' % book.name)
#    return redirect(url_for('show_job', id=id, format=format))

#@app.route('/jobs/new.html')
#def new_job_form():
#    """
#    GET /jobs/new
#
#    Returns a job submission form.
#    """
#    return render_template('new_job.html')

#@app.route('/jobss/<int:id>/edit.html')
#def edit_job_form(id):
#    """
#    GET /books/<id>/edit
#
#    Form for editing job details
#    """
#    book = Book(id=id, name=u'Something crazy') # Your query
#    return render_template('edit_book.html', book=book)

def _transfer_files(jobid):
    logging.warn("XSS attacks possible if stored file is mimetype html")
    for file in flask.request.files.getlist('file'):
        if not file: continue
        filename = secure_filename(os.path.split(file.filename)[1])
        # Because we used named temporary files that aren't deleted on
        # closing as our streaming file type, we can simply move the
        # resulting files to the store rather than copying them.
        file.stream.close()
        logging.warn("moving %s -> %s"%(file.stream.name, os.path.join(store.path(jobid),filename)))
        os.rename(file.stream.name, os.path.join(store.path(jobid),filename))


def init_scheduler(conf):
    if conf == 'slurm':
        from slurm import Scheduler
    elif conf == 'direct':
        logging.warn("direct scheduler is not a good choice!")
        try: os.nice(19)
        except: pass
        from simplequeue import Scheduler
    elif conf == 'dispatch':
        from dispatcher import Scheduler
    else:
        raise ValueError("unknown scheduler %s"%conf)
    return Scheduler()

def serve():
    app.run(host='0.0.0.0')

def fullpath(p): return os.path.abspath(os.path.expanduser(p))
def configure(jobstore=None, jobkey=None, jobdb=None, scheduler=None):
    global SCHEDULER, app

    if jobstore:
        store.ROOT = fullpath(jobstore)
    if jobkey:
        app.config['SECRET_KEY'] = open(fullpath(jobkey)).read()
    if jobdb:
        import jobqueue.db
        jobqueue.db.DB_URI = jobdb

    SCHEDULER = init_scheduler(scheduler)

if __name__ == '__main__':
    configure(jobstore='/tmp/server/%s',
              jobdb='sqlite:///tmp/jobqueue.db',
              jobkey='~/.reflserve/key',
              scheduler='dispatch',
              )
    app.config['DEBUG'] = True
    serve()
