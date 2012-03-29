import wx

IS_MAC = (wx.Platform == '__WXMAC__')

class LogView(wx.Panel):
    title = 'Log'
    default_size = (600,200)
    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        self.log_info = []
        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.progress = wx.TextCtrl(self,-1,style=wx.TE_MULTILINE|wx.HSCROLL)
        self._redraw()

        vsizer.Add(self.progress, 1, wx.EXPAND)

        self.SetSizer(vsizer)
        vsizer.Fit(self)
        self.SetAutoLayout(True)
        #self.SetupScrolling()

        self.Bind(wx.EVT_SHOW, self.OnShow)

    def OnShow(self, event):
        if not event.Show: return
        #print "showing log"
        if self._need_redraw:
            #print "-redraw"
            self._redraw()

    def get_state(self):
        return self.log_info
    def set_state(self, state):
        self.log_info = state
        self._redraw()

    def log_message(self, message):
        if len(self.log_info) > 1000:
            del self.log_info[:-1000]
        self.log_info.append(message)
        self._redraw()

    def _redraw(self):
        if not IS_MAC and not self.IsShown():
            self._need_redraw = True
        else:
            self._need_redraw = False
            self.progress.Clear()
            self.progress.AppendText("\n".join(self.log_info))
