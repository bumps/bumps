'''
Build layout for histogram plots
'''

__all__ = ['build_axes_hist']

from matplotlib import pyplot as plt
from numpy import ceil, sqrt
from os import name as osname

if osname == 'nt':
    fontfamily = 'Arial'
elif osname == 'posix':
    fontfamily = 'sans-serif'

def tile_axes(n):
    """
    Determine number of columns by finding the
    next greatest square, then determine number
    of rows needed.
    """
    
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n/float(cols)))
    return cols, rows

def build_axes_hist(n):
    '''
    Build a figure with one axis per parameter,
    and one axis (the last one) to contain the colorbar.
    Use to make the vars histogram figure.
    '''
    
    #configure the plot parameters
    fontsz = 12
    lwidth = 1
    pad = 2
    pltparams = [['xtick.direction','in'],
                 ['ytick.direction','in'],
                 ['lines.linewidth',lwidth],
                 ['axes.linewidth',lwidth],
                 ['xtick.labelsize',fontsz],
                 ['ytick.labelsize',fontsz],
                 ['xtick.major.size',5],
                 ['ytick.major.size',5],
                 ['xtick.minor.size',2.5],
                 ['ytick.minor.size',2.5],
                 ['xtick.major.width',lwidth],
                 ['ytick.major.width',lwidth],
                 ['xtick.minor.width',lwidth],
                 ['ytick.minor.width',lwidth],
                 ['xtick.major.pad',pad],
                 ['ytick.major.pad',pad],
                 ['xtick.top',True],
                 ['ytick.right',True],
                 ['font.size',fontsz],
                 ['font.family',fontfamily],
                 ['svg.fonttype','none'],
                 ['savefig.dpi',100]]
    for i in pltparams:
        plt.rcParams[i[0]]=i[1]
    
    col, row = tile_axes(n)
    tile_W = 3.0
    tile_H = 2.0
    cbar_width = 0.75

    #set space between plots in horiz and vert
    h_space = 0.2
    v_space = 0.2

    #set top, bottom, left margins
    t_margin = 0.2
    b_margin = 0.2
    l_margin = 0.2
    r_margin = 0.4

    #calculate total width and figure size
    plots_width = (tile_W+h_space)*col
    total_width = plots_width+cbar_width+l_margin+r_margin
    total_height = (tile_H+v_space)*row+t_margin+b_margin
    fsize = [total_width,total_height]

    #calculate dimensions as a faction of figure size
    v_space_f = v_space/total_height
    h_space_f = h_space/total_width

    tile_H_f = tile_H/total_height
    tile_W_f = tile_W/total_width
    cbar_W_f = cbar_width/total_width

    t_margin_f = t_margin/total_height
    b_margin_f = b_margin/total_height
    l_margin_f = l_margin/total_width
    top = 1-t_margin_f+v_space_f
    left = l_margin_f

    #Calculate colorbar location (left,bottom)
    #and colorbar height
    l_cbar_f = l_margin_f+col*(tile_W_f+h_space_f)
    b_cbar_f = b_margin_f+v_space_f
    cbar_H_f = 1 - t_margin_f - b_margin_f - v_space_f
    cbar_box = [l_cbar_f,b_cbar_f,cbar_W_f,cbar_H_f]    

    fig = plt.figure(figsize=fsize)
    k = 0
    for j in range(1,row+1):
        for i in range(0,col):
            if k>=n:
                break
            dims = [left+i*(tile_W_f+h_space_f),
                    top-j*(tile_H_f+v_space_f),
                    tile_W_f,
                    tile_H_f]
            ax=fig.add_axes(dims)
            ax.set_facecolor('none')
            k+=1

    fig.add_axes(cbar_box)
    fig.set_size_inches(total_width,total_height)
    return fig
