#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: ericyuan
  Research on brexit data: sentiment analysis
  Source: ~code/NLP/Brexit/sentiment

@Modified for dashboard by: Minsu Yeom
@On March 19, 2019
"""

import pandas as pd
import numpy as np

# os methods for manipulating paths
from os.path import dirname, join

from bokeh.plotting import figure

from bokeh.models import (CategoricalColorMapper, HoverTool,
						  ColumnDataSource, Panel,
						  FuncTickFormatter, SingleIntervalTicker, LinearAxis,
                          RangeTool)

from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,
								  Tabs, CheckboxButtonGroup,
								  TableColumn, DataTable, Select)

from bokeh.layouts import column, row, layout, gridplot, WidgetBox
from bokeh.palettes import Blues5, Greys5

# List of lists to single list
from itertools import chain


def sentiment_tab():
    # Load "sentiment from headlines", creadted by a separate code
    df_head = pd.read_csv('./dashboard/data/headsentiment.csv',
                        parse_dates=True)
    df_head.reset_index()

    df_body = pd.read_csv('./dashboard/data/bodysentiment.csv',
                        parse_dates=True)
    df_body.reset_index()

    # Join two dataframes into one dataframe using the key, named 'time'
    df = df_head.join(df_body, lsuffix='_head', rsuffix='_body')

    dates = np.array(df['time_head'], dtype=np.datetime64)

    # Get a column of pandas df, return a moving average of that column with window size <- window_size
    def set_moving_avg(df, window_size):
        window = np.ones(window_size)/float(window_size)
        return np.convolve(df.copy(), window, 'same')

    # Set initial window sizes to 10 to both of the head and body.
    mv_subj_head = set_moving_avg(df['subj_head'], 5)
    mv_subj_body = set_moving_avg(df['subj_body'], 5)

    # Create a cds object for headlines
    cds_head = ColumnDataSource(
        data=dict(date=dates, polar_head=df['polar_head'], subj_head=mv_subj_head))

    # Create a cds object for body
    cds_body = ColumnDataSource(
        data=dict(date=dates, polar_body=df['polar_body'], subj_body=mv_subj_body))


    p_polar_head = figure(plot_height=300, plot_width=500, tools="", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above",
            background_fill_color="#efefef", x_range=(dates[-10000], dates[-1]),
            y_range=(-1,1)
            )

    p_subj_head = figure(plot_height=300, plot_width=500, tools="", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above",
            background_fill_color="#efefef", x_range=(dates[-10000], dates[-1]),
            y_range=(0,1)
            )

    p_polar_body = figure(plot_height=300, plot_width=500, tools="", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above",
            background_fill_color="#efefef", x_range=(dates[-10000], dates[-1]),
            y_range=(-1,1)
            )

    p_subj_body = figure(plot_height=300, plot_width=500, tools="", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above",
            background_fill_color="#efefef", x_range=(dates[-10000], dates[-1]),
            y_range=(0,1)
            )
            

    # Setting:  Plot a line chart for polar - News headlines
    p_polar_head.line('date', 'polar_head', source=cds_head, line_color=Greys5[1])
    p_polar_head.title.text = 'Polar in news headlines'
    p_polar_head.yaxis.axis_label = 'Measurements'

    # Setting: Plot a line chart for subjectivity - News headlines
    p_subj_head.line('date', 'subj_head', source=cds_head, line_color=Blues5[0])
    p_subj_head.title.text = 'Subjectivity in news headlines'
    p_subj_head.yaxis.axis_label = 'Measurements'

    # Setting:  Plot a line chart for polar - News bodies
    p_polar_body.line('date', 'polar_body', source=cds_body, line_color=Greys5[1])
    p_polar_body.title.text = 'Polar in news bodies'
    p_polar_body.yaxis.axis_label = 'Measurements'

    # Setting: Plot a line chart for subjectivity - News bodies
    p_subj_body.line('date', 'subj_body', source=cds_body, line_color=Blues5[0])
    p_subj_body.title.text = 'Subjectivity in news bodies'
    p_subj_body.yaxis.axis_label = 'Measurements'

    select_polar_head = figure(title="Drag the middle and edges of the box below to change the range above",
                    plot_height=110, plot_width=500, 
                    y_range=(-1,1),
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="white")

    select_subj_head = figure(title="Drag the middle and edges of the box below to change the range above",
                    plot_height=110, plot_width=500,
                    y_range=p_subj_head.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="white")
    
    select_polar_body = figure(title="",
                    plot_height=110, plot_width=500, 
                    y_range=(-1,1),
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="white")

    select_subj_body = figure(title="",
                    plot_height=110, plot_width=500,
                    y_range=p_subj_head.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="white")

    # Setting: Plot a range tool for polar - News headlines
    range_tool_polar_head = RangeTool(x_range=p_polar_head.x_range)
    range_tool_polar_head.overlay.fill_color = Blues5[2]
    range_tool_polar_head.overlay.fill_alpha = 0.4

    # Setting: Plot a range tool for subjectivity - News headlines
    range_tool_subj_head = RangeTool(x_range=p_subj_head.x_range)
    range_tool_subj_head.overlay.fill_color = Blues5[2]
    range_tool_subj_head.overlay.fill_alpha = 0.4

    # Setting: Plot a range tool for polar - News headlines
    range_tool_polar_body = RangeTool(x_range=p_polar_body.x_range)
    range_tool_polar_body.overlay.fill_color = Blues5[2]
    range_tool_polar_body.overlay.fill_alpha = 0.4

    # Setting: Plot a range tool for subjectivity - News headlines
    range_tool_subj_body = RangeTool(x_range=p_subj_body.x_range)
    range_tool_subj_body.overlay.fill_color = Blues5[2]
    range_tool_subj_body.overlay.fill_alpha = 0.4

    # Plot a range tool for polar - News headlines
    select_polar_head.line('date', 'polar_head', source=cds_head, line_color=Greys5[3])
    select_polar_head.ygrid.grid_line_color = None
    select_polar_head.add_tools(range_tool_polar_head)
    select_polar_head.toolbar.active_multi = range_tool_polar_head

    # Plot a range tool for subjectivity - News headlines
    select_subj_head.line('date', 'subj_head', source=cds_head, line_color=Blues5[1])
    select_subj_head.ygrid.grid_line_color = None
    select_subj_head.add_tools(range_tool_subj_head)
    select_subj_head.toolbar.active_multi = range_tool_subj_head

    # Plot a range tool for polar - News bodies
    select_polar_body.line('date', 'polar_body', source=cds_body, line_color=Greys5[3])
    select_polar_body.ygrid.grid_line_color = None
    select_polar_body.add_tools(range_tool_polar_body)
    select_polar_body.toolbar.active_multi = range_tool_polar_body

    # Plot a range tool for subjectivity - News bodies
    select_subj_body.line('date', 'subj_body', source=cds_body, line_color=Blues5[1])
    select_subj_body.ygrid.grid_line_color = None
    select_subj_body.add_tools(range_tool_subj_body)
    select_subj_body.toolbar.active_multi = range_tool_subj_body

    # Plot sliders 
    slider_mv = Slider(title="Smoothing by N days (moving average)", value=5, start=1, end=20, step=1, width=300)
    #s_subj_body = Slider(title="", value=5, start=1, end=20, step=1)

    # Register event handlers for sliders to get a window size of moving avergaes
    def update_mv(attrname, old, new):
        cds_head.data['subj_head'] = set_moving_avg(df['subj_head'], new)
        cds_body.data['subj_body'] = set_moving_avg(df['subj_body'], new)

    slider_mv.on_change('value', update_mv)


    hover_polar_head = p_polar_head.select(dict(type=HoverTool))
    hover_polar_head.tooltips = [("value", "@polar_head"), ("date", "@date")]
    hover_polar_head.mode = 'mouse'

    l = gridplot(children = [
            [p_polar_head, p_polar_body],
            [select_polar_head, select_polar_body],
            [p_subj_head, p_subj_body],
            [select_subj_head, select_subj_body],
            [None, slider_mv]
            ]
        )

    tab = Panel(child=l, title='Sentiment')

    return tab



