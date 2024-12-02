import io
import csv
from typing import Literal, TypedDict, Any


def csv2dict(csvdata: str) -> dict:
    with io.StringIO(csvdata) as f:
        reader = csv.DictReader(f)
        result = dict(header=reader.fieldnames, rows=[row for row in reader], raw=csvdata)

    return result


def dict2csv(csvdict: dict) -> str:
    # No longer needed since raw csv data are added to the table data
    # dictionary in csv2dict
    with io.StringIO() as f:
        writer = csv.DictWriter(f, fieldnames=csvdict["header"])
        writer.writerows(csvdict["rows"])
        result = f.getvalue()

    return result


class CustomWebviewPlot(TypedDict):
    fig_type: Literal["plotly", "matplotlib", "table", "error"] = "plotly"
    plotdata: Any


def process_custom_plot(plot_item: CustomWebviewPlot) -> CustomWebviewPlot:
    figtype = plot_item["fig_type"]
    plot_data = plot_item["plotdata"]

    if figtype == "plotly":
        figdict = plot_data.to_dict()
    elif figtype == "matplotlib":
        import mpld3

        figdict = mpld3.fig_to_dict(plot_data)
    elif figtype == "table":
        figdict = csv2dict(plot_data)
    elif figtype == "error":
        figdict = plot_data
    else:
        figdict = dict(error=f"unrecognized plot type {figtype}")

    del plot_data  # is this necessary? Does plot_item['plot_data'] also need to be deleted?

    return CustomWebviewPlot(fig_type=figtype, plotdata=figdict)
