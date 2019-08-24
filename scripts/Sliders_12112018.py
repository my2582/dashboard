#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool,BoxSelectTool

output_file("Probability.html")

# Set up data
df = pd.read_csv("/Users/eugene/Google Drive/Columbia Affiliation/Machine Learning for Global Risk/TR Corpus/CombinedProbability_Values.csv")
df['date'] =  pd.to_datetime(df['date'])
x = df['date']
y = 100*df['probability'] + df['pos_neg_diff'] + df['num_news_score'] #记得把function_1改成probability!!
source = ColumnDataSource(data=dict(x=x, y=y))
# N = 200
# x = np.linspace(0, 4*np.pi, N)
# y = x+x**2+x**3
# source = ColumnDataSource(data=dict(x=x, y=y))

# Set up plot
plot = figure(plot_height=400, plot_width=1000, title="Combined Function",
              tools="crosshair,pan,reset,save,wheel_zoom",x_axis_type='datetime')
              #x_range=[0, 4*np.pi], y_range=[-1000, 1000])
# plot = figure(plot_height=400, plot_width=400, title="my comprehensive function",
#               tools="crosshair,pan,reset,save,wheel_zoom",
#               x_range=[0, 4*np.pi], y_range=[-1000, 1000])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# hover = fig1.select(dict(type=HoverTool))
# hover.tooltips = [("Date", "@x{%F}"),  ("Value", "@y"),]
# hover.formatters = {'x':'datetime'}

# Set up widgets
text = TextInput(title="title", value='Combined Function')
weight1 = Slider(title="weight1", value=0.5, start=0.5, end=3.0, step=0.01)
weight2 = Slider(title="weight2", value=0.5, start=0.0, end=10.0, step=0.01)
weight3 = Slider(title="weight3", value=0.5, start=0.0, end=10.0, step=0.01)
#freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)

# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    w_1 = weight1.value
    w_2 = weight2.value
    w_3 = weight3.value
    #k = freq.value

    # Generate the new curve
    x = df['date']
    y = w_1*100*df['probability']+w_2*df['pos_neg_diff']+w_3*df['num_news_score']
    # x = np.linspace(0, 4*np.pi, N)
    # y = (a/(a+b+w))*x+(b/(a+b+w))*x**2+(w/(a+b+w))*x**3

    source.data = dict(x=x, y=y)

for w in [weight1,weight2,weight3]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = widgetbox(text, weight1, weight2, weight3)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"

show(plot)


# In[ ]:




