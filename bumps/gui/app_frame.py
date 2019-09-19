# Copyright (C) 2006-2011, University of Maryland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: James Krycka

"""
This module implements the AppFrame class which creates the main frame of the
GUI for the Bumps application including a basic menu, tool bar, and status bar.
"""

#==============================================================================

import sys

import wx

from .about import (AboutDialog, APP_TITLE, APP_DESCRIPTION, APP_LICENSE,
                    APP_CREDITS, APP_TUTORIAL)
from .app_panel import AppPanel
from .console import NumpyConsole
from .utilities import resource, choose_fontsize, display_fontsize

# Resource files.
PROG_ICON = "bumps.ico"

#==============================================================================
class ModelConsole(NumpyConsole):
    def OnChanged(self, added=(), changed=(), removed=()):
        pass
    def OnClose(self, event):
        self.Show(False)

class AppFrame(wx.Frame):
    """
    This class creates the top-level frame for the application and populates it
    with application specific panels and widgets.
    """

    def __init__(self, parent=None, id=wx.ID_ANY, title=APP_TITLE,
                 pos=wx.DefaultPosition, size=wx.DefaultSize, name="AppFrame"):
        wx.Frame.__init__(self, parent, id, title, pos, size, name=name)

        # Display the application's icon in the title bar.
        icon = wx.Icon(resource(PROG_ICON), wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

        # Set the default font family and font size for the application.
        self.set_default_font()

        # Initialize the menu bar with common items.
        self.add_menubar()

        # Initialize the tool bar.
        self.add_toolbar()

        # Initialize the status bar.
        self.add_statusbar()

        # Build the application panels for the GUI on the frame.
        self.panel = AppPanel(self)
        self.panel.console = ModelConsole(self)
        self.panel.console['app'] = self

        # Note: Do not call self.Fit() as this will reduce the frame to its
        # bare minimum size; we want it to keep its default size.

    def set_default_font(self):
        """
        Sets the default font family and font size for the frame which will be
        inherited by all child windows subsequently created.
        """

        # Save the system default font information before we make any changes.
        default_fontname = self.GetFont().GetFaceName()
        default_fontsize = self.GetFont().GetPointSize()

        # If requested, override the font name to use.  Note that:
        # - the MS Windows default font appears to be the same as Tahoma
        # - Arial tends to be narrower and taller than Tahoma.
        # - Verdana tends to be wider and shorter than Tahoma.
        fontname = default_fontname
        if len(sys.argv) > 1:
            if '--tahoma' in sys.argv[1:]: fontname = "Tahoma"
            if '--arial' in sys.argv[1:]: fontname = "Arial"
            if '--verdana' in sys.argv[1:]: fontname = "Verdana"

        fontsize = choose_fontsize(fontname=fontname)

        # If requested, override the font point size to use.
        if len(sys.argv) > 1:
            if '--12pt' in sys.argv[1:]: fontsize = 12
            if '--11pt' in sys.argv[1:]: fontsize = 11
            if '--10pt' in sys.argv[1:]: fontsize = 10
            if '--9pt' in sys.argv[1:]: fontsize = 9
            if '--8pt' in sys.argv[1:]: fontsize = 8
            if '--7pt' in sys.argv[1:]: fontsize = 7
            if '--6pt' in sys.argv[1:]: fontsize = 6

        # Set the default font for this and all child windows.  The font of the
        # frame's title bar is not affected (which is a good thing).  However,
        # setting the default font does not affect the font used in the frame's
        # menu bar or menu items (which is not such a good thing because the
        # menu text size be different than the size used by the application's
        # other widgets).  The menu font cannot be changed by wxPython.
        self.SetFont(wx.Font(fontsize, wx.SWISS, wx.NORMAL, wx.NORMAL, False,
                             fontname))

        # If requested, display font and miscellaneous platform information.
        if len(sys.argv) > 1 and '--platform' in sys.argv[1:]:
            print("*** Platform =", wx.PlatformInfo)
            print("*** Default font is %s  Chosen font is %s"\
                  %(default_fontname, self.GetFont().GetFaceName()))
            print("*** Default point size = %d  Chosen point size = %d"\
                  %(default_fontsize, self.GetFont().GetPointSize()))
            display_fontsize(fontname=fontname)


    def add_menubar(self):
        """Creates a default menu bar, menus, and menu options."""

        # Create the menu bar.
        mb = wx.MenuBar()
        #wx.MenuBar.SetAutoWindowMenu(False)

        # Add a 'File' menu to the menu bar and define its options.
        file_menu = wx.Menu()

        _item = file_menu.Append(wx.ID_ANY, "&Exit", "Terminate application")
        self.Bind(wx.EVT_MENU, self.OnExit, _item)

        mb.Append(file_menu, "&File")

        # Add a 'Help' menu to the menu bar and define its options.
        help_menu = wx.Menu()

        _item = help_menu.Append(wx.ID_ANY, "&About",
                                            "Get description of application")
        self.Bind(wx.EVT_MENU, self.OnAbout, _item)
        _item = help_menu.Append(wx.ID_ANY, "&Documentation",
                                            "Get User's Guide and Reference Manual")
        self.Bind(wx.EVT_MENU, self.OnTutorial, _item)
        _item = help_menu.Append(wx.ID_ANY, "License",
                                            "Read license and copyright notice")
        self.Bind(wx.EVT_MENU, self.OnLicense, _item)
        _item = help_menu.Append(wx.ID_ANY, "Credits",
                                            "Get list of authors and sponsors")
        self.Bind(wx.EVT_MENU, self.OnCredits, _item)

        help_menu.AppendSeparator()
        _item = help_menu.Append(wx.ID_ANY, "&Console",
                                            "Interactive Python shell")
        self.Bind(wx.EVT_MENU, self.OnConsole, _item)

        mb.Append(help_menu, "&Help")

        # Attach the menu bar to the frame.
        self.SetMenuBar(mb)


    def add_toolbar(self):
        """Creates a default tool bar."""

        #tb = self.CreateToolBar()
        tb = wx.ToolBar(parent=self, style=wx.TB_HORIZONTAL|wx.NO_BORDER)
        tb.Realize()
        self.SetToolBar(tb)


    def add_statusbar(self):
        """Creates a default status bar."""

        sb = self.statusbar = self.CreateStatusBar()
        sb.SetFieldsCount(1)


    def OnAbout(self, evt):
        """Shows the About dialog box."""

        dlg = AboutDialog(parent=self, title="About", info=APP_DESCRIPTION,
                          show_name=True, show_notice=True, show_link=True,
                          show_link_docs=True)
        dlg.ShowModal()
        dlg.Destroy()


    def OnCredits(self, evt):
        """Shows the Credits dialog box."""

        dlg = AboutDialog(parent=self, title="Credits", info=APP_CREDITS,
                          show_name=True, show_notice=True, show_link=False,
                          show_link_docs=False)
        dlg.ShowModal()
        dlg.Destroy()


    def OnExit(self, event):
        """Terminates the program."""
        self.Close()


    def OnLicense(self, evt):
        """Shows the License dialog box."""

        dlg = AboutDialog(parent=self, title="License", info=APP_LICENSE,
                          show_name=True, show_notice=True, show_link=False,
                          show_link_docs=False)
        dlg.ShowModal()
        dlg.Destroy()


    def OnTutorial(self, event):
        """Shows the Tutorial dialog box."""

        dlg = AboutDialog(parent=self, title="Tutorial", info=APP_TUTORIAL,
                          show_name=False, show_notice=False, show_link=False,
                          show_link_docs=True)
        dlg.ShowModal()
        dlg.Destroy()

    def OnConsole(self, event):
        """Raise python console."""
        self.panel.console.Show(True)
