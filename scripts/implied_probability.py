#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Kevin Choi
  Source: ~code/Implied/implied probability brexit case_original .ipynb

@Modified for dashboard by: Minsu Yeom
@On March 24, 2019
"""


import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel, HoverTool, BoxSelectTool, DatetimeTickFormatter
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.layouts import layout, WidgetBox
from bokeh.events import MouseEnter, MouseLeave

def implied_probability_tab():
    # Set up data
    df = pd.read_csv('./dashboard/data/implied_probability.csv', parse_dates=True)
    df['date'] =  pd.to_datetime(df.iloc[:, 0])
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    x = df['date']
    y = df['Implied_prob']*100

    # Store them into bokeh's data frame
    source = ColumnDataSource(data=dict(x=x, y=y))

    # Set up a main plot
    plot = figure(plot_height=400, plot_width=750, title="Implied Probability",
                tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type='datetime')
                #x_range=[0, 4*np.pi], y_range=[-1000, 1000])
        
    # Add hover tooltips
    plot.add_tools(HoverTool(
        tooltips=[
            ( 'date',   '@x{%F}' ),
            ( 'value',  '@y{0.00}%' ),
        ],

        formatters={
            'x' : 'datetime', # use 'datetime' formatter for 'date' field
            'value' : 'printf',   
                                
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    ))

    # Formatting the dates on x-axis
    xformatter = DatetimeTickFormatter(
        days=["%b %d, '%g"],
        months=["%b %Y"],
        years=["%Y"])

    plot.xaxis.formatter = xformatter

    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    def update_data(attrname, old, new):
        # Generate the new curve
        x = df['date']
        y = df['Implied_prob']

        source.data = dict(x=x, y=y)

    # Create a row layout
    # row(WidgetBox, figure, width=#)
    l = layout(children=[plot],
        sizing_mode='fixed'
    )

    # Make a tab with the layout
    tab = Panel(child=l, title="Implied Probability")

    return tab


