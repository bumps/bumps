# This program is public domain
# Author Paul Kienzle
"""
2-D correlation histograms

Generate 2-D correlation histograms and display them in a figure.

Uses false color plots of density.
"""
__all__ = ['Corr2d']

import numpy
from numpy import inf

from matplotlib import cm, colors, image, artist
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

#COLORMAP = cm.gist_stern_r #@UndefinedVariable
#COLORMAP = cm.gist_earth_r #@UndefinedVariable
#COLORMAP = cm.gist_ncar_r #@UndefinedVariable
#COLORMAP = cm.Paired #@UndefinedVariable
try:
    COLORMAP = colors.LinearSegmentedColormap.from_list('density',
                                                        ('w','y','g','b','r'))
except:
    COLORMAP = cm.gist_earth_r #@UndefinedVariable

class Corr2d:
    """
    Generate and manage 2D correlation histograms.
    """
    def __init__(self, data, labels=None, **kw):
        if labels == None:
            labels = ["P"+str(i+1) for i,_ in enumerate(data)]
        self.N = len(data)
        self.labels = labels
        self.data = data
        self.hists = _hists(data, **kw)
        #for k,v in self.hists.items():
        #    print k,(v[1][0],v[1][-1]),(v[2][0],v[2][-1])

    def R(self):
        return numpy.corrcoef(self.data)

    def __getitem__(self, i, j):
        """
        Retrieve correlation histogram for data[i] X data[j].

        Returns bin i edges, bin j edges, and histogram
        """
        return self.hists[i,j]

    def plot(self, title=None):
        """
        Plot the correlation histograms on the specified figure
        """
        import pylab

        fig = pylab.gcf()
        if title != None:
            fig.text(0.5, 0.95, title,
                     horizontalalignment='center',
                     fontproperties=FontProperties(size=16))
        self.ax = _plot(fig, self.hists, self.labels, self.N)

def _hists(data, ranges=None, **kw):
    """
    Generate pair-wise correlation histograms
    """
    N = len(data)
    if ranges == None:
        low,high = numpy.min(data,axis=1), numpy.max(data,axis=1)
        ranges = [(l,h) for l,h in zip(low,high)]
    return dict(((i,j), numpy.histogram2d(data[i], data[j],
                                    range=[ranges[i],ranges[j]], **kw))
                for i in range(0,N)
                for j in range(i+1,N))

def _plot(fig, hists, labels, N, show_ticks=False):
    """
    Plot pair-wise correlation histograms
    """
    vmin, vmax = inf, -inf
    for data,_,_ in hists.values():
        positive = data[data>0]
        if len(positive) > 0:
            vmin = min(vmin,numpy.amin(positive))
            vmax = max(vmax,numpy.amax(positive))
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
    #norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = image.FigureImage(fig)
    mapper.set_array(numpy.zeros(0))
    mapper.set_cmap(cmap=COLORMAP)
    mapper.set_norm(norm)

    ax = {}
    Nr = Nc = N-1
    for i in range(0,N-1):
        for j in range(i+1,N):
            sharex = ax.get((0,j), None)
            sharey = ax.get((i,i+1), None)
            a = fig.add_subplot(Nr,Nc,(Nr-i-1)*Nc + j,
                                sharex=sharex, sharey=sharey)
            ax[(i,j)] = a
            a.xaxis.set_major_locator(MaxNLocator(4,steps=[1,2,4,5,10]))
            a.yaxis.set_major_locator(MaxNLocator(4,steps=[1,2,4,5,10]))
            data,x,y = hists[(i,j)]
            data = numpy.clip(data,vmin,vmax)
            a.pcolorfast(y,x,data,cmap=COLORMAP,norm=norm)
            # Show labels or hide ticks
            if i != 0:
                artist.setp(a.get_xticklabels(),visible=False)
            if i == N-2 and j == N-1:
                a.set_xlabel(labels[j])
                #a.xaxis.set_label_position("top")
                #a.xaxis.set_offset_position("top")
            if not show_ticks:
                a.xaxis.set_ticks([])
            if j == i+1:
                a.set_ylabel(labels[i])
            else:
                artist.setp(a.get_yticklabels(),visible=False)
            if not show_ticks:
                a.yaxis.set_ticks([])

            a.zoomable=True


    # Adjust subplots and add the colorbar
    fig.subplots_adjust(left=.07, bottom=.07, top=.9, right=0.85,
                        wspace=0.0, hspace=0.0)
    cax = fig.add_axes([0.88, 0.2, 0.04, 0.6])
    fig.colorbar(mapper, cax=cax, orientation='vertical')
    return ax

def zoom(event,step):
    ax = event.inaxes
    if not hasattr(ax,'zoomable'): return

    # TODO: test logscale
    step = 3*step
    def rescale(lo,hi,pt,step):
        if step > 0: scale = float(hi-lo)*step/100
        else: scale = float(hi-lo)*step/(100-step)
        bal = float(pt-lo)/(hi-lo)
        lo = lo - bal*scale
        hi = hi + (1-bal)*scale
        return (lo,hi)


    if ax.zoomable is not True and 'mapper' in ax.zoomable:
        mapper = ax.zoomable['mapper']
        if event.ydata is not None:
            lo, hi = mapper.get_clim()
            pt = event.ydata*(hi-lo)+lo
            lo, hi = rescale(lo,hi,pt,step)
            mapper.set_clim((lo,hi))
    if ax.zoomable is True and event.xdata is not None:
        lo, hi = ax.get_xlim()
        lo, hi = rescale(lo,hi,event.xdata,step)
        ax.set_xlim((lo,hi))
    if ax.zoomable is True and event.ydata is not None:
        lo, hi = ax.get_ylim()
        lo, hi = rescale(lo,hi,event.ydata,step)
        ax.set_ylim((lo,hi))
    ax.figure.canvas.draw_idle()
