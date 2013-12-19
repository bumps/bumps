# This code is in the public domain
# Author: Paul Kienzle

# TODO: better cache control; some requests shouldn't be cached
# TODO: compression
# TODO: streaming file transfer
# TODO: single/multifile response objects?

# Inspirations:
#   Upload files in python:
#     http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/146306
#   urllib2_file:
#     Fabien Seisen: <fabien@seisen.org>
#   MultipartPostHandler:
#     Will Holcomb <wholcomb@gmail.com>
#   python-rest-client
#     Benjamin O'Steen
from six import StringIO
from six.moves.urllib import parse

import email
import httplib2
import mimetypes
import uuid



DEFAULT_CONTENT="application/octet-stream"
class Connection(object):
    def __init__(self, url, username=None, password=None):
        self.url = url
        http = httplib2.Http() #".cache")
        http.follow_all_redirects = True
        if username and password:
            http.add_credentials(username, password)
        self.url  = url
        self.http = http

    def get(self, resource, fields={}):
        return _request(self.http, 'GET', self.url+resource, fields=fields)
    def head(self, resource, fields={}):
        return _request(self.http, 'HEAD', self.url+resource, fields=fields)
    def post(self, resource, fields={}, body=None, mimetype=DEFAULT_CONTENT):
        return _request(self.http, 'POST', self.url+resource, fields=fields,
                        body=body, mimetype=mimetype)
    def put(self, resource, fields={}, body=None, mimetype=DEFAULT_CONTENT):
        return _request(self.http, 'PUT', self.url+resource, fields=fields,
                        body=body, mimetype=mimetype)
    def postfiles(self, resource, fields={}, files=None):
        return _request(self.http, 'POST', self.url+resource,
                        fields=fields, files=files)
    def putfiles(self, resource, fields={}, files=None):
        return _request(self.http, 'PUT', self.url+resource,
                        fields=fields, files=files)
    def delete(self, resource, fields={}):
        return _request(self.http, 'DELETE', self.url+resource, fields=fields)

def _request(http, verb, location, fields=None,
             body=None, mimetype=None, files=None):

    headers = {'User-Agent': 'Basic Agent'}
    if files:
        if body:
            raise TypeError("Use fields instead of body with file upload")
        # Note: this section is public domain; the old code wasn't working
        boundary = uuid.uuid4().hex
        buf = StringIO()
        for key,value in fields.items():
            buf.write(u'--%s\r\n'%boundary)
            buf.write(u'Content-Disposition: form-data; name="%s"' % key)
            buf.write(u'\r\n\r\n%s\r\n'%value)
        for key,filename in enumerate(files):
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            buf.write(u'--%s\r\n'%boundary)
            buf.write(u'Content-Disposition: form-data; name="file"; filename="%s"\r\n' % filename)
            buf.write(u'Content-Type: %s\r\n\r\n' % content_type)
            buf.write(open(filename,'rb').read())
            buf.write(u'\r\n')
        buf.write(u'--%s--\r\n'%boundary)
        body = buf.getvalue()
        headers['Content-Type'] = 'multipart/form-data; boundary='+boundary
        headers['Content-Length'] = str(len(body))
        #print "===== body =====\n",body
    elif body:
        if fields:
            raise TypeError("Body, if included, should encode fields directly.")
        headers['Content-Type']=mimetype
        headers['Content-Length'] = str(len(body))
    elif fields:
        if verb == "GET":
            location += u'?' + parse.urlencode(fields)
            body = u''
        else:
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
            body = parse.urlencode(fields)

    #print "uri",location
    #print "body",body
    #print "headers",headers
    #print "method",verb
    try:
        response, content = http.request(location, verb,
                                         body=body, headers=headers)
    except AttributeError:
        raise IOError("Could not open "+location)

    return response, content.decode('UTF-8')


# Table mapping response codes to messages; entries have the
# form {code: (shortmessage, longmessage)}.
RESPONSE = {
    100: ('Continue', 'Request received, please continue'),
    101: ('Switching Protocols',
          'Switching to new protocol; obey Upgrade header'),

    200: ('OK', 'Request fulfilled, document follows'),
    201: ('Created', 'Document created, URL follows'),
    202: ('Accepted',
          'Request accepted, processing continues off-line'),
    203: ('Non-Authoritative Information', 'Request fulfilled from cache'),
    204: ('No Content', 'Request fulfilled, nothing follows'),
    205: ('Reset Content', 'Clear input form for further input.'),
    206: ('Partial Content', 'Partial content follows.'),

    300: ('Multiple Choices',
          'Object has several resources -- see URI list'),
    301: ('Moved Permanently', 'Object moved permanently -- see URI list'),
    302: ('Found', 'Object moved temporarily -- see URI list'),
    303: ('See Other', 'Object moved -- see Method and URL list'),
    304: ('Not Modified',
          'Document has not changed since given time'),
    305: ('Use Proxy',
          'You must use proxy specified in Location to access this '
          'resource.'),
    307: ('Temporary Redirect',
          'Object moved temporarily -- see URI list'),

    400: ('Bad Request',
          'Bad request syntax or unsupported method'),
    401: ('Unauthorized',
          'No permission -- see authorization schemes'),
    402: ('Payment Required',
          'No payment -- see charging schemes'),
    403: ('Forbidden',
          'Request forbidden -- authorization will not help'),
    404: ('Not Found', 'Nothing matches the given URI'),
    405: ('Method Not Allowed',
          'Specified method is invalid for this server.'),
    406: ('Not Acceptable', 'URI not available in preferred format.'),
    407: ('Proxy Authentication Required', 'You must authenticate with '
          'this proxy before proceeding.'),
    408: ('Request Timeout', 'Request timed out; try again later.'),
    409: ('Conflict', 'Request conflict.'),
    410: ('Gone',
          'URI no longer exists and has been permanently removed.'),
    411: ('Length Required', 'Client must specify Content-Length.'),
    412: ('Precondition Failed', 'Precondition in headers is false.'),
    413: ('Request Entity Too Large', 'Entity is too large.'),
    414: ('Request-URI Too Long', 'URI is too long.'),
    415: ('Unsupported Media Type', 'Entity body in unsupported format.'),
    416: ('Requested Range Not Satisfiable',
          'Cannot satisfy request range.'),
    417: ('Expectation Failed',
          'Expect condition could not be satisfied.'),

    500: ('Internal Server Error', 'Server got itself in trouble'),
    501: ('Not Implemented',
          'Server does not support this operation'),
    502: ('Bad Gateway', 'Invalid responses from another server/proxy.'),
    503: ('Service Unavailable',
          'The server cannot process the request due to a high load'),
    504: ('Gateway Timeout',
          'The gateway server did not receive a timely response'),
    505: ('HTTP Version Not Supported', 'Cannot fulfill request.'),
    }
