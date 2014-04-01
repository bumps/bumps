# ======
# Put current directory and sasview directory on path.
# This won't be necessary once bumps is in sasview
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import periodictable
except ImportError:
    from distutils.util import get_platform
    platform = '.%s-%s'%(get_platform(),sys.version[:3])
    pt = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..','..','periodictable'))
    sys.path.insert(0,pt)
try: 
    import sans
except ImportError:
    from distutils.util import get_platform
    platform = '.%s-%s'%(get_platform(),sys.version[:3])
    sasview = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..','..','sasview','build','lib'+platform))
    sys.path.insert(0,sasview)
    #raise Exception("\n".join(sys.path))
# ======

from sasbumps import *

M = load_fit('FitPage2.fitv')
problem = FitProblem([M])

