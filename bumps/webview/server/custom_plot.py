from typing import Literal, TypedDict, Any

class CustomWebviewPlot(TypedDict):
    fig_type: Literal['plotly'] | Literal ['matplotlib'] | Literal['error'] = 'plotly'
    plotdata: Any

def process_custom_plot(plot_item: CustomWebviewPlot) -> CustomWebviewPlot:
    
    figtype = plot_item['fig_type']
    plot_data = plot_item['plotdata']

    if figtype == 'plotly':
        figdict = plot_data.to_dict()
    elif figtype == 'matplotlib':
        import mpld3
        figdict = mpld3.fig_to_dict(plot_data)
    elif figtype == 'error':
        figdict = plot_data
    else:
        figdict = dict(error=f'unrecognized plot type {figtype}')
    
    del plot_data # is this necessary? Does plot_item['plot_data'] also need to be deleted?
    
    return CustomWebviewPlot(fig_type=figtype, plotdata=figdict)

