#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: 
  Source: ~code/Visualizations/Sliders_12112018.ipynb

@Modified for dashboard by: Minsu Yeom
@On March 19, 2019
"""


import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel, HoverTool, BoxSelectTool, DatetimeTickFormatter
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.layouts import layout, WidgetBox
from bokeh.events import MouseEnter, MouseLeave

def combined_probability_tab():
    # Set up data
    # Read data of implied probability
    df_im = pd.read_csv('./dashboard/data/implied_probability.csv', parse_dates=True)
    df_im['date'] =  pd.to_datetime(df_im.iloc[:, 0])
    df_im.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Read data of sentiment analysis and get 
    df_st = pd.read_csv('./dashboard/data/result_sentiment_topic.csv', parse_dates=True)
    df_st.index = pd.Series(df_st['time']).apply(lambda x : pd.to_datetime(x))
    df_news_cnt = df_st.iloc[:, 0:1]
    df_news_cnt = df_news_cnt.to_period(freq='d').groupby('time').count()

    # Create a column whose type is datetime for convenient coding.
    df_news_cnt['date'] = df_news_cnt.index.to_timestamp()

    create_csv(df_news_cnt)

    # df is the data frame to be plotted finally.
    df = df_im.merge(df_news_cnt, on='date', how='left')
    df.fillna(0, inplace=True)

    # 'cumul_no' is the cumulative number of articles for the past 30 days. (rolling)
    df['cumul_no'] = df['No'].rolling(window=30).sum()
    df.fillna(0, inplace=True)

    # STANDARDIZE: news_score is currently defined by (X - mean) / stddev(X) for the whole dataset
    df['news_score']=(df['cumul_no'] - df['cumul_no'].mean()) / df['cumul_no'].std()

    # Count the number of positive articles and the number of negative articles.
    df_st['pos_cnt'] = df_st['sentiment_logit'].eq(4)*1
    df_st['neg_cnt'] = df_st['sentiment_logit'].eq(0)*1
    df_pos_neg_cnt = df_st[['pos_cnt', 'neg_cnt']].to_period(freq='d').groupby('time').sum()
    df_pos_neg_cnt.head()

    # Cumulative the number of pos' and neg' articles for the past 30 days. (rolling)
    df_cum_pos_neg_cnt = df_pos_neg_cnt.rolling(window=30).sum()
    df_cum_pos_neg_cnt.fillna(0, inplace=True)
    df_cum_pos_neg_cnt['cum_diff_pos_neg'] = df_cum_pos_neg_cnt['pos_cnt'] - df_cum_pos_neg_cnt['neg_cnt']
    df_cum_pos_neg_cnt['date'] = df_cum_pos_neg_cnt.index.to_timestamp()

    # STANDARDIZE: news_score is currently defined by (X - mean) / stddev(X) for the whole dataset
    df_cum_pos_neg_cnt['diff_pos_neg_score']=(df_cum_pos_neg_cnt['cum_diff_pos_neg'] - df_cum_pos_neg_cnt['cum_diff_pos_neg'].mean()) / df_cum_pos_neg_cnt['cum_diff_pos_neg'].std()

    # Mergr it with df
    df = df.merge(df_cum_pos_neg_cnt, on='date', how='left')
    df.fillna(0, inplace=True)

    #df = pd.read_csv("./dashboard/data/CombinedProbability_Values.csv")
    #df['date'] =  pd.to_datetime(df['date'])

    x = df['date']
    y = 100*df['Implied_prob'] + df['diff_pos_neg_score'] + df['news_score'] 
    
    # Set up data of components
    y_prob = 100*df['Implied_prob']
    y_pos_neg_diff = df['diff_pos_neg_score'] 
    y_num_news_score = df['news_score']

    # Store them into bokeh's data frame
    source = ColumnDataSource(data=dict(x=x, y=y))
    source_prob = ColumnDataSource(data=dict(x=x, y=y_prob))
    source_pos_neg_diff = ColumnDataSource(data=dict(x=x, y=y_pos_neg_diff))
    source_num_news_score = ColumnDataSource(data=dict(x=x, y=y_num_news_score))


    # Set up a main plot
    plot = figure(plot_height=500, plot_width=1200, title="Combined Function",
                tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type='datetime', toolbar_location=None)

    # Add hover tooltips
    hover = HoverTool(
        tooltips=[
            ('date',   '@x{%F}'),
            ('value',  '@y{0.00}%'),
        ],

        formatters={
            'x': 'datetime',  # use 'datetime' formatter for 'date' field
            'value': 'printf',

        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    )
    plot.add_tools(hover)

    # Formatting the dates on x-axis
    xformatter = DatetimeTickFormatter(
        days=["%b %d, '%g"],
        months=["%b %Y"],
        years=["%Y"])

    plot.xaxis.formatter = xformatter

    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    # Set up plots of the components of the main plot.
    # Component 1 - probability
    plot_w1 = figure(plot_height=300, plot_width=500, title="Implied Probability",
                tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type='datetime', toolbar_location=None)

    plot_w1.line('x', 'y', source=source_prob, line_width=1, line_alpha=0.6)

    # Component 2 - difference between the number of positives and negatives
    plot_w2 = figure(plot_height=300, plot_width=500, title="Polarity",
                tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type='datetime', toolbar_location=None)

    plot_w2.line('x', 'y', source=source_pos_neg_diff, line_width=1, line_alpha=0.6)

    # Component 3 - the number of news scores
    plot_w3 = figure(plot_height=300, plot_width=500, title="News Flow",
                tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type='datetime', toolbar_location=None)

    plot_w3.line('x', 'y', source=source_num_news_score, line_width=1, line_alpha=0.6)

    # Add hover tooltips
    hover_unitless = HoverTool(
        tooltips=[
            ('date',   '@x{%F}'),
            ('value',  '@y{0.00}'),
        ],

        formatters={
            'x': 'datetime',  # use 'datetime' formatter for 'date' field
            'value': 'printf',

        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline'
    )


    plot_w1.add_tools(hover)
    plot_w2.add_tools(hover_unitless)
    plot_w3.add_tools(hover_unitless)


    # Set up widgets
    text = TextInput(value='Combined Function')
    weight1 = Slider(title="Weight 1. Implied Probability", value=0.5, start=0.5, end=3.0, step=0.01)
    weight2 = Slider(title="Weight 2. Polarity", value=0.5, start=0.0, end=10.0, step=0.01)
    weight3 = Slider(title="Weight 3. News Flow", value=0.5, start=0.0, end=10.0, step=0.01)

    # Set up callbacks
    def update_title(attrname, old, new):
        plot.title.text = text.value

    text.on_change('value', update_title)

    def update_data(attrname, old, new):
        # Get the current slider values
        w_1 = weight1.value
        w_2 = weight2.value
        w_3 = weight3.value

        # Generate the new curve
        x = df['date']
        y = w_1*100*df['Implied_prob']+w_2*df['diff_pos_neg_score']+w_3*df['news_score']

        y_prob = 100*df['Implied_prob']
        y_pos_neg_diff = df['diff_pos_neg_score']
        y_num_news_score = df['news_score']

        source.data = dict(x=x, y=y)
        source_prob.data = dict(x=x, y=y_prob)
        source_pos_neg_diff.data = dict(x=x, y=y_pos_neg_diff)
        source_num_news_score.data = dict(x=x, y=y_num_news_score)

    def callback(event):
        print("name, values: ", event.event_name, event.event_values)
        

    for w in [weight1, weight2, weight3]:
        w.on_change('value', update_data)
    
    weight1.on_event(MouseEnter, callback)
    weight1.on_event(MouseLeave, callback)
        # w.x_range.on_event(MouseLeave, callback)
        # w.x_range.on_event(MouseEnter, callback)

    # Set up layouts and add to document
    inputs = WidgetBox(text, weight1, weight2, weight3)

    # Create a row layout
    # row(WidgetBox, figure, width=#)
    l = layout(children=[
        [plot, inputs],
        [plot_w1, plot_w2, plot_w3]
        ],
        sizing_mode='fixed'
    )


    # Make a tab with the layout
    tab = Panel(child=l, title="Combined Probability")

    return tab

def create_csv(df_news_cnt):
    idx = pd.date_range(df_news_cnt['date'].min(), df_news_cnt['date'].max())
    df_filled = df_news_cnt.set_index('date').reindex(idx).fillna(0).rename_axis('date').reset_index()
    df_filled.to_csv('news_count.csv')