# Pandas for data management
import pandas as pd

# os methods for manipulating paths
from os.path import dirname, join

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs


# Each tab is drawn by one script
from scripts.histogram import histogram_tab
from scripts.density import density_tab
from scripts.table import table_tab
from scripts.routes import route_tab


# Done.
from scripts.sentiment import sentiment_tab
from scripts.word_cloud import word_cloud_tab
from scripts.market_indicator import market_indicator_tab
from scripts.combined_probability import combined_probability_tab


# Using included state data from Bokeh for map
from bokeh.sampledata.us_states import data as states

# Read data into dataframes
flights = pd.read_csv(join(dirname(__file__), 'data', 'flights.csv'), 
	                                          index_col=0).dropna()

# Formatted Flight Delay Data for map
map_data = pd.read_csv(join(dirname(__file__), 'data', 'flights_map.csv'),
                            header=[0,1], index_col=0)

# Create each of the tabs
tab1 = sentiment_tab()
tab2 = word_cloud_tab()
#tab3 = market_indicator_tab()
tab4 = combined_probability_tab()

# Put all the tabs into one application
tabs = Tabs(tabs = [tab1,
                    tab2,
 #                   tab3,
                    tab4])

# Put the tabs in the current document for display
curdoc().add_root(tabs)




