# This program is in the public domain
# Author: Paul Kienzle
"""
Substitution references in hyperlinks.

In order to construct documents programmatically with references to
version specific download files for example, you will need to be able
to control the generation of the text from the configure script.

For this purpose we provide the substitution link, or slink, role to
sphinx.  Within conf.py you must define *slink_vars*, which is a
dictionary of variables which can be used for substitution.  Within
your RST documents, you can then use :slink:`pattern` with standard
python 2.x string substition template rules.  The pattern is usually
"text <url>" but "url" defaults to "url <url>" with proper html escapes.

For example::

    -- conf.py --
    ...
    extensions.append('slink')
    slink_vars = dict(url="http://some.url.com",
                      source="sputter-%s.zip"%version,
                      )
    ...

    -- download.rst --
    ...
    Source: :slink:`latest sputter <%(url)s/downloads/%(source)s>`
    ...
"""

import traceback
from docutils import nodes, utils


def setup(app):
    def slink_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
        def warn(err):
            msg = "\n  error in %s\n  %s" % (rawtext, err)
            inliner.reporter.warning(msg, line=lineno)

        try:
            text = text % app.config.slink_vars
        except Exception as exc:
            # err = traceback.format_exc(0).strip()
            err = traceback.format_exception_only(exc.__class__, exc)[0]
            warn(err.strip())
        lidx, ridx = text.find("<"), text.find(">")
        if lidx >= 0 and ridx > lidx and ridx == len(text) - 1:
            ref = text[lidx + 1 : ridx]
            name = utils.unescape(text[:lidx].strip())
        elif lidx > 0 or ridx > 0:
            warn("Incorrect reference format in expanded link: " + text)
            ref = ""
            name = utils.unescape(text)
        else:
            ref = text
            name = utils.unescape(ref)
        node = nodes.reference(rawtext, name, refuri=ref, **options)
        return [node], []

    app.add_config_value("slink_vars", {}, False)
    app.add_role("slink", slink_role)
