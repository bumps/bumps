"""A special directive for including wx panels.

Given a path to a .py file, it includes the source code inline, and an
image of the panel it produces.

This directive supports all of the options of the `image` directive,
except for `target` (since plot will add its own target).

Additionally, if the :include-source: option is provided, the literal
source will be included inline, as well as a link to the source.

The set of file formats to generate can be specified with the
plot_formats configuration variable.
"""

# Note: adapted from matplotlib.sphinxext.plot_directive by Paul Kienzle

from six.moves import StringIO

import sys, os, glob, shutil, hashlib, imp, warnings
import re
try:
    from hashlib import md5
except ImportError:
    from md5 import md5
from docutils.parsers.rst import directives
try:
    # docutils 0.4
    from docutils.parsers.rst.directives.images import align
except ImportError:
    # docutils 0.5
    from docutils.parsers.rst.directives.images import Image
    align = Image.align
from docutils import nodes
import sphinx

import wx
# Matplotlib helper utilities
import matplotlib.cbook as cbook
import numpy as np

from . import png


sphinx_version = sphinx.__version__.split(".")
# The split is necessary for sphinx beta versions where the string is
# '6b1'
sphinx_version = tuple([int(re.split('[a-z]', x)[0])
                        for x in sphinx_version[:2]])


if hasattr(os.path, 'relpath'):
    relpath = os.path.relpath
else:
    def relpath(target, base=os.curdir):
        """
        Return a relative path to the target from either the current dir or an optional base dir.
        Base can be a directory specified either as absolute or relative to current dir.
        """

        if not os.path.exists(target):
            raise OSError('Target does not exist: '+target)

        if not os.path.isdir(base):
            raise OSError('Base is not a directory or does not exist: '+base)

        base_list = (os.path.abspath(base)).split(os.sep)
        target_list = (os.path.abspath(target)).split(os.sep)

        # On the windows platform the target may be on a completely different drive from the base.
        if os.name in ('nt','dos','os2') and base_list[0] != target_list[0]:
            raise OSError('Target is on a different drive to base. Target: '+target_list[0].upper()+', base: '+base_list[0].upper())

        # Starting from the filepath root, work out how much of the filepath is
        # shared by base and target.
        for i in range(min(len(base_list), len(target_list))):
            if base_list[i] != target_list[i]: break
        else:
            # If we broke out of the loop, i is pointing to the first differing path elements.
            # If we didn't break out of the loop, i is pointing to identical path elements.
            # Increment i so that in all cases it points to the first differing path elements.
            i+=1

        rel_list = [os.pardir] * (len(base_list)-i) + target_list[i:]
        return os.path.join(*rel_list)

def write_char(s):
    sys.stdout.write(s)
    sys.stdout.flush()

options = {'alt': directives.unchanged,
           'height': directives.length_or_unitless,
           'width': directives.length_or_percentage_or_unitless,
           'scale': directives.nonnegative_int,
           'align': align,
           'class': directives.class_option,
           'include-source': directives.flag }

template = """
.. image:: %(prefix)s%(tmpdir)s/%(outname)s
   %(options)s
"""

exception_template = """
.. htmlonly::

   [`source code <%(linkdir)s/%(basename)s.py>`__]

Exception occurred rendering plot.

"""

def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.
    """
    return (not os.path.exists(derived))
    # or os.stat(derived).st_mtime < os.stat(original).st_mtime)

def runfile(fullpath):
    """
    Import a Python module from a path.
    """
    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.
    pwd = os.getcwd()
    path, fname = os.path.split(fullpath)
    sys.path.insert(0, os.path.abspath(path))
    stdout = sys.stdout
    sys.stdout = StringIO()
    os.chdir(path)
    try:
        fd = open(fname)
        module = imp.load_module("__main__", fd, fname, ('py', 'r', imp.PY_SOURCE))
    finally:
        del sys.path[0]
        os.chdir(pwd)
        sys.stdout = stdout
    return module

def capture_image(panel, labels):
    # Need to be at a top level window in order to force a redraw
    frame = panel
    while not frame.IsTopLevel():
        frame = frame.parent
    frame.Show()
    wx.Yield()

    # Grab the bitmap; if it is the top level, then include WindowDC so we
    # can grab the window decorations.  This only works on Windows!
    if panel.IsTopLevel():
        graphdc = wx.WindowDC(panel)
    else:
        graphdc = wx.ClientDC(panel)
    w,h = graphdc.GetSize()
    bmp = wx.EmptyBitmap(w,h)
    memdc = wx.MemoryDC()
    memdc.SelectObject(bmp)
    memdc.Blit(0,0, w, h, graphdc, 0, 0)

    # Add annotations using a GCDC so we get antialiased corners
    gcdc = wx.GCDC(memdc)
    for widget,label,position in labels:
        annotate(gcdc, widget=widget, label=label, position=position,
                 panelsize=(w,h))

    # Release the bitmap from the DC
    memdc.SelectObject(wx.NullBitmap)

    # Copy bitmap to a numpy array
    img = np.empty((w,h,3),'uint8')
    bmp.CopyToBuffer(buffer(img), format=wx.BitmapBufferFormat_RGB)

    # Destroy the frame
    frame.Destroy()
    wx.Yield()
    return img

def write_png(outpath,img):
    w,h,p = img.shape
    img = np.ascontiguousarray(img)
    writer = png.Writer(size=(w,h), alpha=False, bitdepth=8, compression=9)
    with open(outpath,'wb') as fid:
        writer.write(fid, np.reshape(img,(h,w*p)))

def annotate(dc, widget, label, position='c', panelsize=(0,0)):
    """
    Draws label relative to the widget on the panel.

    *panel* is the panel to receive the annotation
    *widget* is the widget or coordinates (x,y) in panel to be annotated
    *label* is the annotation label
    *position* is the location of the annotation, which is one of:
        * t: above the widget
        * b: below the widget
        * l: left of the widget
        * r: right of the widget
        * c: center of the widget
    """
    padx, pady = 4,4        # Space around rectangle
    bordersize = 2          # Size of border line
    fontsize = 18           # Size of text
    radius = (fontsize+pady+bordersize)//2   # Rounding radius on rectangle
    marginx, marginy = 2,2  # Space be edge rectangle and edge of widget
    foreground = 'black'    # Font and outline colour
    background = '#C1A004C0'  # Gold fill

    pen = wx.Pen(colour=foreground, width=bordersize)
    brush = wx.Brush(colour=background)
    font = wx.Font(pointSize=fontsize,
                   family=wx.FONTFAMILY_SWISS,
                   style=wx.FONTSTYLE_NORMAL,
                   weight=wx.FONTWEIGHT_NORMAL
                   )
    dc.SetPen(pen)
    dc.SetBrush(brush)
    dc.SetFont(font)

    # Determine box dimensions
    tw,th = dc.GetTextExtent(label)
    rw,rh = tw+2*padx,th+2*pady

    # If the box is tall and thin, force it to be a circle because it looks
    # better.  Conveniently, numbers 1-9 as annotations should all be circles.
    # TODO: maybe draw this as a circle rather than rounded rectangle?
    if rw < rh:
        padx += (rh-rw)//2
        rw = rh

    # Determine anchor position on the screen, which is either the
    # rectangle containing a specific widget, or is a pair of coordinates (x,y)
    try:     # Is it (x,y)?
        bx,by = widget
        bw,bh = 0,0
    except:  # No.  Hope it is a widget
        bx,by = widget.GetPositionTuple()
        bw,bh = widget.GetSizeTuple()

    # Position the label relative to the anchor
    if position == 't':
        rx = bx + (bw-rw)//2
        ry = by - (marginy + rh)
    elif position == 'b':
        rx = bx + (bw-rw)//2
        ry = by + bh + marginy
    elif position == 'l':
        rx = bx - (marginx + rw)
        ry = by + (bh-rh)//2
    elif position == 'r':
        rx = bx + bw + marginx
        ry = by + (bh-rh)//2
    elif position == 'c':
        rx = bx + (bw-rw)//2
        ry = by + (bh-rh)//2
    else:
        raise ValueError('position should be t, l, b, r, or c')

    # Make sure label box doesn't fall off the panel
    #fw,fh = dc.GetSize()
    fw,fh = panelsize # Grrr... antialiasing DC does not preserve size
    #print "*** text",label,tw,th
    #print " ** widget",bx,by,bw,bh
    #print " ** rect",rx,ry,rw,rh
    #print " ** panel",fw,fh
    if rx+rw >= fw: rx = fw-(rw+bordersize//2 + 1)
    if ry+rh >= fh: ry = fh-(rh+bordersize//2 + 1)
    if rx < 0:   rx = bordersize//2
    if ry < 0:   ry = bordersize//2

    # Draw the box and the annotation label
    dc.BeginDrawing()
    dc.DrawRoundedRectangle(rx,ry,rw,rh,radius)
    dc.DrawText(text=label,x=rx+padx,y=ry+pady)
    dc.EndDrawing()

def make_image(fullpath, code, outdir, context='', options={}):
    """
    run a script and save the PNG in _static
    """

    fullpath = str(fullpath)  # todo, why is unicode breaking this
    basedir, fname = os.path.split(fullpath)
    basename, ext = os.path.splitext(fname)

    if str(basename) == "None":
        import pdb
        pdb.set_trace()

    # Look for output file
    outpath = os.path.join(outdir, basename+'.png')
    if not out_of_date(fullpath, outpath):
        write_char('.')
        return 1

    # We didn't find the files, so build them

    if code is not None:
        exec(code)
    else:
        try:
            module = runfile(fullpath)
            panel = module.panel
        except:
            warnings.warn("current path "+os.getcwd())
            s = cbook.exception_to_str("Exception running wx %s %s" % (fullpath,context))
            warnings.warn(s)
            return False

    try:    labels
    except: labels = []
    img = capture_image(panel,labels)
    write_png(outpath, img)

    return True

def wx_directive(name, arguments, options, content, lineno,
                   content_offset, block_text, state, state_machine):
    """
    Handle the plot directive.
    """
    # The user may provide a filename *or* Python code content, but not both
    if len(arguments) == 1:
        reference = directives.uri(arguments[0])
        basedir, fname = os.path.split(reference)
        basename, ext = os.path.splitext(fname)
        basedir = relpath(basedir, setup.app.builder.srcdir)
        if len(content):
            raise ValueError("wx directive may not specify both a filename and inline content")
        content = None
    else:
        basedir = "inline"
        content = '\n'.join(content)
        # Since we don't have a filename, use a hash based on the content
        reference = basename = md5(content).hexdigest()[-10:]
        fname = None

    # Get the directory of the rst file, and determine the relative
    # path from the resulting html file to the plot_directive links
    # (linkdir).  This relative path is used for html links *only*,
    # and not the embedded image.  That is given an absolute path to
    # the temporary directory, and then sphinx moves the file to
    # build/html/_images for us later.
    rstdir, rstfile = os.path.split(state_machine.document.attributes['source'])
    reldir = rstdir[len(setup.confdir)+1:]
    relparts = [p for p in os.path.split(reldir) if p.strip()]
    nparts = len(relparts)
    outdir = os.path.join('wx_directive', basedir)
    linkdir = ('../' * nparts) + outdir

    context = "at %s:%d"%(rstfile,lineno)

    # tmpdir is where we build all the output files.  This way the
    # plots won't have to be redone when generating latex after html.

    # Prior to Sphinx 0.6, absolute image paths were treated as
    # relative to the root of the filesystem.  0.6 and after, they are
    # treated as relative to the root of the documentation tree.  We need
    # to support both methods here.
    tmpdir = os.path.join('_build', outdir)
    if sphinx_version < (0, 6):
        tmpdir = os.path.abspath(tmpdir)
        prefix = ''
    else:
        prefix = '/'
    if not os.path.exists(tmpdir):
        cbook.mkdirs(tmpdir)

    # destdir is the directory within the output to store files
    # that we'll be linking to -- not the embedded images.
    destdir = os.path.abspath(os.path.join(setup.app.builder.outdir, outdir))
    if not os.path.exists(destdir):
        cbook.mkdirs(destdir)

    # Generate the figures, and return the number of them
    success = make_image(reference, content, tmpdir, context=context,
                         options=options)

    if 'include-source' in options:
        if content is None:
            content = open(reference, 'r').read()
        lines = ['::', ''] + ['    %s'%row.rstrip() for row in content.split('\n')]
        del options['include-source']
    else:
        lines = []

    if success:
        options = ['      :%s: %s' % (key, val) for key, val in options.items()]
        options = "\n".join(options)
        if fname is not None:
            try:
                shutil.copyfile(reference, os.path.join(destdir, fname))
            except:
                s = cbook.exception_to_str("Exception copying plot %s %s" % (reference,context))
                warnings.warn(s)
                return 0

        outname = basename+'.png'

        # Copy the linked-to files to the destination within the build tree,
        # and add a link for them
        shutil.copyfile(os.path.join(tmpdir, outname),
                        os.path.join(destdir, outname))

        # Output the resulting reST
        lines.extend((template % locals()).split('\n'))
    else:
        lines.extend((exception_template % locals()).split('\n'))

    if len(lines):
        state_machine.insert_input(
            lines, state_machine.input_lines.source(0))

    return []

def setup(app):
    global _WXAPP
    _WXAPP = wx.PySimpleApp()

    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_directive('wx', wx_directive, True, (0, 1, 0), **options)
