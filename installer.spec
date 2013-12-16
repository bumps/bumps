# -*- mode: python -*-
import sys
import os

import bumps
print "found bumps in %r"%bumps.__file__
version = str(bumps.__version__)

excludes = ['IPython.html','IPython.nbconvert','IPython.qt','IPython.testing',
            'sphinx','docutils','jinja2',
            ]
a = Analysis(['bin/bumps_gui'],
             pathex=[],
             hookspath=['extra/installer-hooks'],
             excludes=excludes,
             runtime_hooks=None)
#print "\n".join("%s: %s"%(f[-1],", ".join(f[:-1])) for f in a.datas)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='Bumps %s'%version,
          debug=False,
          strip=None,
          upx=True,
          console=False , icon='extra/bumps.icns')
app = BUNDLE(exe,
             name='Bumps %s.app'%version,
             icon='extra/bumps.icns')
