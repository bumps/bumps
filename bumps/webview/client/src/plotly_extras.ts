import * as Plotly from "plotly.js/lib/core";

export const SVGDownloadButton = {
  name: "Download as SVG",
  title: "Download as SVG",
  icon: Plotly.Icons["camera-retro"],
  click: function (gd: Plotly.RootOrData) {
    Plotly.downloadImage(gd, {
      format: "svg",
      width: null,
      height: null,
      filename: "",
    });
  },
};

export const PNGDownloadButton = {
  name: "Download plot as PNG",
  title: "Download plot as PNG",
  icon: Plotly.Icons.camera, // default icon for PNG download
  click: function (gd: Plotly.RootOrData) {
    Plotly.downloadImage(gd, {
      format: "png",
      width: null,
      height: null,
      filename: "",
    });
  },
};

// A reusable Plotly config that includes the SVG download button
// and moves the default PNG download button to the end.
export const configWithSVGDownloadButton: Partial<Plotly.Config> = {
  modeBarButtonsToRemove: ["toImage"],
  modeBarButtonsToAdd: [PNGDownloadButton, SVGDownloadButton],
};
