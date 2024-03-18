import * as Plotly from 'plotly.js/lib/core';

export const SVGDownloadButton = {
  name: 'Download as SVG',
  title: 'Download as SVG',
  icon: Plotly.Icons.camera,
  click: function(gd) {
    Plotly.downloadImage(gd, {format: 'svg'})
  }
}

export const configWithSVGDownloadButton = {
    modeBarButtonsToAdd: [
      SVGDownloadButton      
    ]
}