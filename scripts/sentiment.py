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
                          RangeTool, DatetimeTickFormatter)

from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,
								  Tabs, CheckboxButtonGroup,
								  TableColumn, DataTable, Select)

from bokeh.layouts import column, row, layout, gridplot, WidgetBox
from bokeh.palettes import Blues5, Greys5

# List of lists to single list
from itertools import chain


def sentiment_tab():
    df = pd.read_csv('./dashboard/data/result_sentiment_topic.csv', parse_dates=True)
    df.index = pd.Series(df['time']).apply(lambda x : pd.to_datetime(x))
    df.drop(['time'], axis=1, inplace=True)

    def get_aggregation(df, dominant_topic_no=99, agg_freq='w', agg_method=1):
        '''
        dominant_topic_no:
            '(1) Trade talks':0,    # 1.eu, 2.britain, 3.europiean, 4.union, 5.market, 6.single*, 7.trade*, 13.bloc*, 14.customs*, 28.custom
            '(2) F/X market':1,     # 1.bank, 2.percent, 4.rate*, 5.economy*, 6.sterling*, 9.boe, 12.euro, 15.fall, 16.rise, 29.inflation*
            '(3) Politics':2,       # 1.party*, 2.vote, 4.minister, 5.parliament*, 6.labour*, 7.government, 8.conservative, 9.theresa, 10.deal
            '(4) UK Economy':3,     # 1.london, 2.financial*, 3.britain, 4.bank, 5.european, 6.business, 8.company, 9.service*, 13.base*, 14.industry*, 25.job
            '(5) Deals':4,          # 1.deal, 2.eu, 3.british, 4.european, 5.agreement, 6.border, 7.irish. 8.ireland, 9.no-deal, 10.union
            '(6) Short story':5                 # 1.brexit*, 2.eu, 3.uk*, 4.britain, 5.hard, 6.london, 7.vote, 8.news*, 9.bank, 10.pound, 11.deal
            '(7) European stock markets':6,     #1.european, 2.percent, 3.market, 5.stock, 6.share, 7.europe, 8.company, 13.ftst, 14.sale, 15.top, 17.profit, 18.open, 21.investor
            '(8) Global financial markets':7,   #1.percent, 2.usd, 3.high, 4.market, 5.low, 6.weak, 7.gdp, 8.trade, 9.dollar, 10.china, 11.fed, 13.risk, 14.index, 15.stock, 18., 19.euro}
        
        agg_freq:
        d - daily frequency
        w - weekly frequency
        m - monthly frequency
        
        agg_method: defines how to aggregate values of either of polar or subjectivity.
                    Any aggregation is calculated over the period selected by 'freq'
        1 - Positive proportion := the number of positive values divided by the total counts
        2 - Negative proportion := the number of negative values divided by the total counts
        3 - Weighted sum := sum of all values divided by the total counts
        4 - % difference between positives and negatives := agg_method 1 minus agg_method 2
        '''
        
        # Filter dataset conditioning on the topic number, i.e., dominant_topic_no.
        if dominant_topic_no != 99 :
            df_filtered = df.loc[df['Dominant_Topic'] == dominant_topic_no].copy()
        else:
            # no conditioning on topic number (i.e. all topics included)
            df_filtered = df.copy()
        
        # Convering date into period of dates and group it by the chosen frequency.
        df_grouped = df_filtered.to_period(freq=agg_freq).groupby('time')
        
        # Define a function that calculate a positive proportion.
        pos_ratio_0 = lambda x: (np.sum(x > 0) / np.size(x))     # for polar
        pos_ratio_5 = lambda x: (np.sum(x > 0.5) / np.size(x))   # for subjectivity
        
        # Define a function that calculate a negative proportion.
        neg_ratio_0 = lambda x: (np.sum(x < 0) / np.size(x))
        neg_ratio_5 = lambda x: (np.sum(x < 0.5) / np.size(x))
        
        # Define a function that calculate a weighted sum
        weighted_sum = lambda x: (np.sum(x) / np.size(x))
        
        if agg_method == 1:
            df_polar = df_grouped['body_polar'].agg(pos_ratio_0)
            df_subj = df_grouped['body_subj'].agg(pos_ratio_5)
            pass
        elif agg_method == 2:
            df_polar = df_grouped['body_polar'].agg(neg_ratio_0)
            df_subj = df_grouped['body_subj'].agg(neg_ratio_5)
            pass
        elif agg_method == 3:
            df_polar = df_grouped['body_polar'].agg(weighted_sum)
            df_subj = df_grouped['body_subj'].agg(weighted_sum)
            pass
        elif agg_method == 4:
            df_polar = df_grouped['body_polar'].agg(pos_ratio_0) - df_grouped['body_polar'].agg(neg_ratio_0)
            df_subj = df_grouped['body_subj'].agg(pos_ratio_5) - df_grouped['body_subj'].agg(neg_ratio_5)
            pass
        else:
            raise Exception("You should choose an aggregation method by setting one of 1, 2, 3 and 4 only to agg_method parameter.")
        
        return df_polar, df_subj


    # Set defaults values for combo boxes at the bottom.
    df_polar, df_subj = get_aggregation(df, 99, 'w', 1)
    dates = df_polar.to_timestamp('w').index

    # Create a cds object for body
    cds_body = ColumnDataSource(
        data=dict(date=dates, polar_body=pd.Series(df_polar), subj_body=pd.Series(df_subj)))


    p_polar_body = figure(plot_height=300, plot_width=500, tools="", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above",
            background_fill_color="#efefef", x_range=(dates[-int(dates.size/2)], dates[-1]),
            y_range=(0, 1)
            )

    p_subj_body = figure(plot_height=300, plot_width=500, tools="", toolbar_location=None,
            x_axis_type="datetime", x_axis_location="above",
            background_fill_color="#efefef", x_range=(dates[-int(dates.size/2)], dates[-1]),
            y_range=(0, 1)
            )

    # Add hover tooltips
    hover_polar = HoverTool(
        tooltips=[
            ('date',   '@date{%F}'),
            ('value',  '@polar_body{0.00}'),
        ],

        formatters={
            'date': 'datetime',  # use 'datetime' formatter for 'date' field
            'value': 'printf',

        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    )

    hover_subj = HoverTool(
        tooltips=[
            ('date',   '@date{%F}'),
            ('value',  '@subj_body{0.00}'),
        ],

        formatters={
            'date': 'datetime',  # use 'datetime' formatter for 'date' field
            'value': 'printf',

        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    )

    p_polar_body.add_tools(hover_polar)
    p_subj_body.add_tools(hover_subj)


    # Formatting the dates on x-axis
    xformatter = DatetimeTickFormatter(
        days=["%b %d, '%g"],
        months=["%b '%g"],
        years=["%Y"])

    p_polar_body.xaxis.formatter = xformatter
    p_subj_body.xaxis.formatter = xformatter

    # Setting:  Plot a line chart for polar - News bodies
    p_polar_body.line('date', 'polar_body', source=cds_body, line_color=Greys5[1])
    p_polar_body.title.text = 'Polarity in news bodies'
    p_polar_body.yaxis.axis_label = 'Measurements'

    # Setting: Plot a line chart for subjectivity - News bodies
    p_subj_body.line('date', 'subj_body', source=cds_body, line_color=Blues5[0])
    p_subj_body.title.text = 'Subjectivity in news bodies'
    p_subj_body.yaxis.axis_label = 'Measurements'

    select_polar_body = figure(title="",
                    plot_height=110, plot_width=500, 
                    y_range=(-1,1),
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="white")

    select_subj_body = figure(title="",
                    plot_height=110, plot_width=500,
                    y_range=(0,1),
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="white")

    # Setting: Plot a range tool for polar - News headlines
    range_tool_polar_body = RangeTool(x_range=p_polar_body.x_range)
    range_tool_polar_body.overlay.fill_color = Blues5[2]
    range_tool_polar_body.overlay.fill_alpha = 0.4

    # Setting: Plot a range tool for subjectivity - News headlines
    range_tool_subj_body = RangeTool(x_range=p_subj_body.x_range)
    range_tool_subj_body.overlay.fill_color = Blues5[2]
    range_tool_subj_body.overlay.fill_alpha = 0.4

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

    topic_names = [
        'All',
        '(1) Trade talks',
        '(2) F/X market',
        '(3) Politics',
        '(4) UK Economy',
        '(5) Deals',
        '(6) Short story',
        '(7) European stock markets',
        '(8) Global financial markets'
        ]

    # We manually decide topic names.
    # Most relevant terms are listed for each topic name in ranking order.
    # "*" refers to a word dominantly appearing in that specific topic out of all topics (more than 2/3 by eye-balling)
    topic_name_dic = {'All': 99,
        '(1) Trade talks':0,    # 1.eu, 2.britain, 3.europiean, 4.union, 5.market, 6.single*, 7.trade*, 13.bloc*, 14.customs*, 28.custom
        '(2) F/X market':1,     # 1.bank, 2.percent, 4.rate*, 5.economy*, 6.sterling*, 9.boe, 12.euro, 15.fall, 16.rise, 29.inflation*
        '(3) Politics':2,       # 1.party*, 2.vote, 4.minister, 5.parliament*, 6.labour*, 7.government, 8.conservative, 9.theresa, 10.deal
        '(4) UK Economy':3,     # 1.london, 2.financial*, 3.britain, 4.bank, 5.european, 6.business, 8.company, 9.service*, 13.base*, 14.industry*, 25.job
        '(5) Deals':4,          # 1.deal, 2.eu, 3.british, 4.european, 5.agreement, 6.border, 7.irish. 8.ireland, 9.no-deal, 10.union
        '(6) Short story':5,                 # 1.brexit*, 2.eu, 3.uk*, 4.britain, 5.hard, 6.london, 7.vote, 8.news*, 9.bank, 10.pound, 11.deal
        '(7) European stock markets':6,     #1.european, 2.percent, 3.market, 5.stock, 6.share, 7.europe, 8.company, 13.ftst, 14.sale, 15.top, 17.profit, 18.open, 21.investor
        '(8) Global financial markets':7   #1.percent, 2.usd, 3.high, 4.market, 5.low, 6.weak, 7.gdp, 8.trade, 9.dollar, 10.china, 11.fed, 13.risk, 14.index, 15.stock, 18., 19.euro
        }
    agg_freq = ['(1) Daily', '(2) Weekly', '(3) Monthly', '(4) Quarterly']
    agg_freq_dic = {'(1) Daily':'d', '(2) Weekly':'w', '(3) Monthly':'m', '(4) Quarterly':'q'}
    agg_method = ['(1) Positive proportion', '(2) Negative proportion', '(3) Weighted sum', '(4) Net polarity']
    agg_method_dic = {'(1) Positive proportion':1, '(2) Negative proportion':2, '(3) Weighted sum':3, '(4) Net polarity':4}

    def update_select(attr, old, new):
        # Origin and destination determine values displayed
        dominant_topic_no = topic_name_dic[topic_name_select.value]
        agg_freq = agg_freq_dic[agg_freq_select.value]
        agg_method = agg_method_dic[agg_method_select.value]

        # Get the new dataset
        polar, subj = get_aggregation(df, dominant_topic_no, agg_freq, agg_method)
        dates = polar.to_timestamp(agg_freq).index

        if agg_method == 1 or agg_method == 2:
            p_polar_body.y_range.start = 0
            p_subj_body.y_range.start = 0
        else:
            p_polar_body.y_range.start = -1
            p_subj_body.y_range.start = -1

        cds_body.data.update(ColumnDataSource(
            data=dict(date=dates, polar_body=pd.Series(polar), subj_body=pd.Series(subj))).data)

        polar.to_csv("body_polarity.csv")
        subj.to_csv("body_subjectivity.csv")

            
    topic_name_select = Select(title = 'Dominant topics about Brexit', value = 'All', options = topic_names)
    topic_name_select.on_change('value', update_select)

    agg_freq_select = Select(title = 'Aggregation frequency', value = '(2) Weekly', options = agg_freq)
    agg_freq_select.on_change('value', update_select)

    agg_method_select = Select(title = 'Aggregation method', value = '(1) Positive proportion', options = agg_method)
    agg_method_select.on_change('value', update_select)


    l = layout(children = [
            [p_polar_body, p_subj_body],
            [select_polar_body, select_subj_body],
            [topic_name_select, agg_freq_select, agg_method_select]
            ],
            sizing_mode='fixed'
        )

    tab = Panel(child=l, title='Sentiment')

    return tab



