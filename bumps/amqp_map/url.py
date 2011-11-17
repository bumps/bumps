# This program is public domain
"""
Parse URLs
"""

import urllib

class URL(object):
    """
    Parse a universal resource locator

        protocol://user:password@host:port/path?p1=v1&p2=v2

    :Parameters:
        *url*  : string
            URL to be parsed
        *host* = 'localhost' : string
            Default host
        *user*, *password*, *protocol*, *path* = '' : string
            Defaults user, password, protocol, path
        *parameters* = []: [ (string,string ), ... ]
            Default key,value pairs for POST queries.
        *port* = 0 : integer
            Default port

    :Returns:
        *url* : URL
            The return URL, with attributes for the pieces

    :Raises:
        *ValueError* : Not a valid protocol
    """
    def __init__(self, url, user='', password='', host='localhost',
                 protocol='', path='', port=0, parameters=[]):
        errmsg = "".join( ("Invalid url <",url,">") )

        # chop protocol
        pieces = url.split('://')
        if len(pieces) == 1:
            url = pieces[0]
        elif len(pieces) == 2:
            protocol = pieces[0]
            url = pieces[1]
        else:
            raise ValueError(errmsg)

        pos = url.find('/')
        if pos < 0:
            server = url
        else:
            server = url[:pos]
            path = url[pos+1:]

        if '@' in server:
            user,server = server.split('@')
            if ':' in user:
                user,password = user.split(':')
        if ':' in server:
            server,port = server.split(':')
            port = int(port)
        if server != '': host = server

        if '?' in path:
            path, pars = path.split('?')
            parameters = [pair.split('=') for pair in pars.split('&')]
            if any(len(pair) > 2 for pair in parameters):
                raise ValueError(errmsg)
            parameters = [[urllib.unquote_plus(p) for p in pair]
                          for pair in parameters]
        self.protocol = protocol
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.path = urllib.unquote_plus(path)
        self.parameters = parameters[:]

    def __str__(self):
        result = []
        if self.protocol:
            result.extend( (self.protocol,'://') )
        if self.password:
            result.extend( (self.user,':',self.password,'@') )
        elif self.user:
            result.extend( (self.user,'@') )
        if self.host:
            result.extend( (self.host,) )
        if self.port:
            result.extend( (':', str(self.port)) )
        if self.path or len(self.parameters) > 0:
            result.extend( ('/', urllib.quote_plus(self.path)) )
        if len(self.parameters) > 0:
            pars = '&'.join('='.join(urllib.quote_plus(p) for p in parts)
                            for parts in self.parameters)
            result.extend( ('?', pars) )
        return ''.join(result)


def test():
    h = URL('a')
    assert h.user=='' and h.password=='' and h.host=='a' and h.port==0
    h = URL('a:4')
    assert h.user=='' and h.password=='' and h.host=='a' and h.port==4
    h = URL('u@a')
    assert h.user=='u' and h.password=='' and h.host=='a' and h.port==0
    h = URL('u:p@a')
    assert h.user=='u' and h.password=='p' and h.host=='a' and h.port==0
    h = URL('u@a:4')
    assert h.user=='u' and h.password=='' and h.host=='a' and h.port==4
    h = URL('u:p@a:4')
    assert h.user=='u' and h.password=='p' and h.host=='a' and h.port==4
    h = URL('')
    assert h.user=='' and h.password=='' and h.host=='localhost' and h.port==0
    h = URL('u@')
    assert h.user=='u' and h.password=='' and h.host=='localhost' and h.port==0
    h = URL('u@:4')
    assert h.user=='u' and h.password=='' and h.host=='localhost' and h.port==4
    h = URL('u:p@')
    assert h.user=='u' and h.password=='p' and h.host=='localhost' and h.port==0
    h = URL('u:p@:4')
    assert h.user=='u' and h.password=='p' and h.host=='localhost' and h.port==4
    h = URL('proto://u:p@:4')
    assert (h.protocol=='proto' and h.user=='u' and h.password=='p'
            and h.host=='localhost' and h.port==4)
    h = URL('proto://u:p@:4/')
    assert (h.protocol=='proto' and h.user=='u' and h.password=='p'
            and h.host=='localhost' and h.port==4 and h.path == '')
    h = URL('proto://u:p@:4/%7econnolly')
    assert (h.protocol=='proto' and h.user=='u' and h.password=='p'
            and h.host=='localhost' and h.port==4 and h.path == '~connolly')
    h = URL('proto://u:p@:4/%7econnolly?this&that=other')
    assert (h.protocol=='proto' and h.user=='u' and h.password=='p'
            and h.host=='localhost' and h.port==4 and h.path == '~connolly')
    assert (h.parameters[0][0] == 'this'
            and h.parameters[1][0] == 'that'
            and h.parameters[1][1] == 'other')
    assert str(h) == 'proto://u:p@localhost:4/%7Econnolly?this&that=other'

if __name__ == "__main__": test()
