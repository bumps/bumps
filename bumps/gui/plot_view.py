import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as Toolbar

# The Figure object is used to create backend-independent plot representations.
from matplotlib.figure import Figure

from .util import EmbeddedPylab

IS_MAC = wx.Platform == "__WXMAC__"


class PlotView(wx.Panel):
    title = "Plot"
    default_size = (600, 400)

    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        # Can specify name on
        if "title" in kw:
            self.title = kw["title"]

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

        self._calculating = False
        self._need_plot = self._need_newmodel = False
        self.Bind(wx.EVT_SHOW, self.OnShow)
        self.plot_state = None
        self.model = None

        """
        # Add context menu and keyboard support to canvas
        canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnContextMenu)
        #canvas.Bind(wx.EVT_LEFT_DOWN, lambda evt: canvas.SetFocus())


        # Status bar
        frame = self.GetTopLevelParent()
        self.statusbar = frame.GetStatusBar()
        if self.statusbar is None:
            self.statusbar = frame.CreateStatusBar()
        status_update = lambda msg: self.statusbar.SetStatusText(msg)

        canvas.mpl_connect('motion_notify_event', self.OnMotion),
        """

    '''
    def OnContextMenu(self,event):
        """
        Forward the context menu invocation to profile, if profile exists.
        """
        transform = self.axes.transData
        sx,sy = event.GetX(), event.GetY()
        data_x,data_y = pixel_to_data(transform, sx, self.fig.bbox.height-sy)

        popup = wx.Menu()
        item = popup.Append(wx.ID_ANY,'&Grid on/off', 'Toggle grid lines')
        wx.EVT_MENU(self, item.GetId(),
                    lambda _: (self.axes.grid(),self.fig.canvas.draw_idle()))
        self.PopupMenu(popup, (sx,sy))
        return False

    def update_cursor(self, x, y):
        def nice(value, range):
            place = int(math.log10(abs(range[1]-range[0]))-3)
            #print value,range,place
            if place<0: return "%.*f"%(-place,value)
            else: return "%d"%int(value)
        self.status_update("x:%s  y:%s"
                           %( nice(x, self.axes.get_xlim()),
                              nice(y, self.axes.get_ylim())))

    def OnMotion(self, event):
        """Respond to motion events by changing the active layer."""

        # Force data coordinates for the mouse position
        transform = self.axes.transData
        x,y = pixel_to_data(transform, event.x, event.y)
        self.update_cursor(x,y)
    '''

    def OnShow(self, event):
        # print "theory show"
        if not event.Show:
            return
        if self._need_newmodel:
            self._redraw(newmodel=True)
        elif self._need_plot:
            self._redraw(newmodel=False)

    def set_model(self, model):
        self.model = model
        if not IS_MAC and not self.IsShown():
            self._need_newmodel = True
        else:
            self._redraw(newmodel=True)

    def update_model(self, model):
        # print "profile update model"
        if self.model != model:  # ignore updates to different models
            return

        if not IS_MAC and not self.IsShown():
            self._need_newmodel = True
        else:
            self._redraw(newmodel=True)

    def update_parameters(self, model):
        # print "profile update parameters"
        if self.model != model:
            return

        if not IS_MAC and not self.IsShown():
            self._need_plot = True
        else:
            self._redraw(newmodel=self._need_newmodel)

    def _redraw(self, newmodel=False):
        self._need_newmodel = newmodel
        if self._calculating:
            # That means that I've entered the thread through a
            # wx.Yield for the currently executing redraw.  I need
            # to cancel the running thread and force it to start
            # the calculation over.
            self.cancel_calculation = True
            # print "canceling calculation"
            return

        # print("plotting", self.title)
        with self.pylab_interface as pylab:
            self._calculating = True

            # print "calling again"
            while True:
                # print "restarting"
                # We are restarting the calculation, so clear the reset flag
                self.cancel_calculation = False

                if self._need_newmodel:
                    self.newmodel()
                    if self.cancel_calculation:
                        continue
                    self._need_newmodel = False
                self.plot()
                if self.cancel_calculation:
                    continue
                pylab.draw()
                break
        self._need_plot = False
        self._calculating = False

    def get_state(self):
        # print "returning state",self.model,self.plot_state
        return self.model, self.plot_state

    def set_state(self, state):
        self.model, self.plot_state = state
        # print "setting state",self.model,self.plot_state
        self.plot()

    def menu(self):
        """
        Return a model specific menu
        """
        return None

    def newmodel(self, model=None):
        """
        New or updated model structure.  Do any sort or precalculation you
        need.  plot will be called separately when you are done.

        For long calculations, periodically perform wx.YieldIfNeeded()
        and then if self.cancel_calculation is True, return from the plot.
        """
        pass

    def plot(self):
        """
        Plot to the current figure.  If model has a plot method,
        just use that.

        For long calculations, periodically perform wx.YieldIfNeeded()
        and then if self.cancel_calculation is True, return from the plot.
        """
        if hasattr(self.model, "plot"):
            self.model.plot()
        else:
            raise NotImplementedError("PlotPanel needs a plot method")
