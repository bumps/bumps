import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as Toolbar

# The Figure object is used to create backend-independent plot representations.
from matplotlib.figure import Figure
from numpy import inf

from ..fitproblem import FitProblem
from .util import EmbeddedPylab

# Can't seem to detect when notebook should be drawn on Mac
IS_MAC = wx.Platform == "__WXMAC__"


# ------------------------------------------------------------------------
class DataView(wx.Panel):
    title = "Data"
    default_size = (600, 400)

    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        # Instantiate a figure object that will contain our plots.
        figure = Figure(figsize=(1, 1), dpi=72)

        # Initialize the figure canvas, mapping the figure object to the plot
        # engine backend.
        canvas = FigureCanvas(self, wx.ID_ANY, figure)

        # Wx-Pylab magic ...
        # Make our canvas an active figure manager for pylab so that when
        # pylab plotting statements are executed they will operate on our
        # canvas and not create a new frame and canvas for display purposes.
        # This technique allows this application to execute code that uses
        # pylab stataments to generate plots and embed these plots in our
        # application window(s).  Use _activate_figure() to set.
        self.pylab_interface = EmbeddedPylab(canvas)

        # Instantiate the matplotlib navigation toolbar and explicitly show it.
        mpl_toolbar = Toolbar(canvas)
        mpl_toolbar.Realize()

        # Create a vertical box sizer to manage the widgets in the main panel.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, border=0)
        sizer.Add(mpl_toolbar, 0, wx.EXPAND | wx.ALL, border=0)

        # Associate the sizer with its container.
        self.SetSizer(sizer)
        sizer.Fit(self)

        self._need_redraw = False
        self.Bind(wx.EVT_SHOW, self.OnShow)
        self._calculating = False
        self.toolbar = mpl_toolbar
        self.view = "linear"

    def menu(self):
        # Add 'View' menu to the menu bar and define its options.
        # Present y-axis plotting scales as radio buttons.
        # Grey out items that are not currently implemented.
        frame = wx.GetTopLevelParent(self)
        menu = wx.Menu()
        _item = menu.AppendRadioItem(wx.ID_ANY, "Li&near", "Plot y-axis in linear scale")
        _item.Check(True)
        frame.Bind(wx.EVT_MENU, self.OnLinear, _item)
        _item = menu.AppendRadioItem(wx.ID_ANY, "&Log", "Plot y-axis in log scale")
        frame.Bind(wx.EVT_MENU, self.OnLog, _item)

        menu.AppendSeparator()

        _item = menu.Append(wx.ID_ANY, "&Residuals", "Show residuals on plot panel")
        frame.Bind(wx.EVT_MENU, self.OnResiduals, _item)
        menu.Enable(id=_item.GetId(), enable=True)

        return menu

    # ==== Views ====
    # TODO: can probably parameterize the view selection.
    def OnLog(self, event):
        self.view = "log"
        self.redraw()

    def OnLinear(self, event):
        self.view = "linear"
        self.redraw()

    def OnResiduals(self, event):
        self.view = "residual"
        self.redraw()

    # ==== Model view interface ===
    def OnShow(self, event):
        # print "theory show"
        if not event.Show:
            return
        # print "showing theory"
        if self._need_redraw:
            # print "-redraw"
            self.redraw()

    def get_state(self):
        return self.problem

    def set_state(self, state):
        self.set_model(state)

    def set_model(self, model):
        self.problem = model
        self.redraw(reset=True)

    def update_model(self, model):
        if self.problem == model:
            self.redraw()

    def update_parameters(self, model):
        if self.problem == model:
            self.redraw()

    # =============================

    def redraw(self, reset=False):
        # Hold off drawing until the tab is visible
        if not IS_MAC and not self.IsShown():
            self._need_redraw = True
            return
        # print "drawing theory"

        if self._calculating:
            # That means that I've entered the thread through a
            # wx.Yield for the currently executing redraw.  I need
            # to cancel the running thread and force it to start
            # the calculation over.
            self._cancel_calculate = True
            # print "canceling calculation"
            return

        self._need_redraw = False
        self._calculating = True

        # Calculate theory
        # print "calling again"
        while True:
            # print "restarting"
            # We are restarting the calculation, so clear the reset flag
            self._cancel_calculate = False

            # clear graph and exit if problem is not defined
            if self.problem is None:
                with self.pylab_interface as pylab:
                    pylab.clf()  # clear the canvas
                    break

            # Preform the calculation
            if isinstance(self.problem, FitProblem):
                # print "n=",len(self.problem.models)
                for p in self.problem.models:
                    self._precalc(p)
                    # print "cancel",self._cancel_calculate,"reset",p.updating
                    if self._cancel_calculate:
                        break
                if self._cancel_calculate:
                    continue
            else:
                self._precalc(self.problem)
                if self._cancel_calculate:
                    continue

            # Redraw the canvas with newly calculated theory
            # TODO: drawing is 10x too slow!
            with self.pylab_interface as pylab:
                ax = pylab.gca()
                # print "reset",reset, ax.get_autoscalex_on(), ax.get_xlim()
                reset = reset or ax.get_autoscalex_on()
                xrange = ax.get_xlim()
                # print "composing"
                pylab.clf()  # clear the canvas
                # shift=20 if self.view == 'log' else 0
                shift = 0
                if isinstance(self.problem, FitProblem):
                    for i, p in enumerate(self.problem.models):
                        # if hasattr(p.fitness,'plot'):
                        p.plot(view=self.view)
                        if self._cancel_calculate:
                            break
                    pylab.text(0.01, 0.01, "chisq=%s" % self.problem.chisq_str(), transform=pylab.gca().transAxes)
                    if self._cancel_calculate:
                        continue
                else:
                    # if hasattr(self.problem.fitness,'plot'):
                    self.problem.plot(view=self.view)
                    if self._cancel_calculate:
                        continue

                # print "drawing"
                if not reset:
                    self.toolbar.push_current()
                    set_xrange(pylab.gca(), xrange)
                    self.toolbar.push_current()
                pylab.draw()
                # print "done drawing"
                break

        self._calculating = False

    def _precalc(self, problem):
        """
        Calculate each model separately, hopefully not blocking the gui too long.
        Individual problems may want more control, e.g., between computing theory
        and resolution.
        """
        _ = problem.nllf()
        wx.Yield()


def set_xrange(ax, xrange):
    miny, maxy = inf, -inf
    for L in ax.get_lines():
        x, y = L.get_data()
        idx = (x > xrange[0]) & (x < xrange[1])
        if idx.any():
            miny = min(miny, min(y[idx]))
            maxy = max(maxy, max(y[idx]))
    if miny < maxy:
        if ax.get_yscale() == "linear":
            padding = 0.05 * (maxy - miny)
            miny, maxy = miny - padding, maxy + padding
        else:
            miny, maxy = miny * 0.95, maxy * 1.05
    ax.set_xlim(xrange)
    ax.set_ylim(miny, maxy)
