#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: unknown. It's from wordcloud2.ipynb in JupyterLab running in AWS.

@Modified for dashboard by: Minsu Yeom
@On March 30, 2019
"""


import skimage
import skimage.io

from bokeh.plotting import figure, show
from bokeh.layouts import layout
from bokeh.models import Panel


def word_cloud_tab():
    def to_bokeh_image(im, plot_width=400, plot_height=600):
        '''
        im = an instane of skimage.io.imread()
        e.g. im = skimage.io.imread('./wordcloud8.png')
        '''

        n, m, k = im.shape

        if plot_width is None:
            plot_width = int(m/n*plot_height)
        
        if plot_height is None:
            plot_height = int(n/m*plot_width)
        
        p = figure(x_range=(0, m), y_range=(0, n),
                plot_width=plot_width, plot_height=plot_height, toolbar_location=None)
        p.image_rgba(image=[im[::-1, :]], x=0, y=0, dw=m, dh=n)

        return p

    im = skimage.io.imread('./dashboard//data/wordcloud8_new_24_large.png')

    p = to_bokeh_image(im, plot_width=1600, plot_height=800)
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.visible = False
    p.outline_line_color = None

    # Create a row layout
    # row(WidgetBox, figure, width=#)
    l = layout(children=[p],
        sizing_mode='fixed'
    )

    # Make a tab with the layout
    tab = Panel(child=l, title="Word Cloud")

    return tab


