#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jiajing Wang and Quenue
  Source: ~code/Outliers/MarketIndex2019.ipynb

@Modified for dashboard by: Minsu Yeom
@On March 19, 2019
"""

import igraph as ig
import datetime
import pandas as pd
import numpy as np

from bokeh.plotting import figure

from bokeh.models import (CategoricalColorMapper, HoverTool, BoxSelectTool,
						  ColumnDataSource, Panel,
						  FuncTickFormatter, SingleIntervalTicker, LinearAxis,
                          RangeTool)

from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,
								  Tabs, CheckboxButtonGroup,
								  TableColumn, DataTable, Select)

from bokeh.layouts import column, row, layout, WidgetBox
from bokeh.palettes import Blues5, Greys5

# List of lists to single list
from itertools import chain


def market_indicator_tab():
    # data = pd.read_excel(
    #     '/Users/msyeom/Google Drive/Machine Learning for Global Risk/Market Data/IndexData_new.xlsx', sheet_name='Valuefrom 2000')
    data = pd.read_excel(
        './dashboard/data/IndexData_new.xlsx', sheet_name='Valuefrom 2000')
    data = pd.DataFrame(data)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.dropna(subset=['FTSE Italia All-Share '])

    # In[3]:

    data = data[['Date', 'FTSE 100 INDEX ', 'DAX INDEX ', 'CAC 40 INDEX ', 'AEX-Index ', 'FTSE Italia All-Share ', 'BEL 20 INDEX ',
                 'Euro Stoxx 50 Pr ', 'Austrian Traded Index', 'IBEX 35', 'PSI 20']]
    data = data.rename(columns={'FTSE 100 INDEX ': 'FTSE_100_Index', 'DAX INDEX ': 'DAX_Index', 'CAC 40 INDEX ': 'CAC_40_Index',
                                'AEX-Index ': 'AEX_Index', 'FTSE Italia All-Share ': 'FTSE_Italia', 'BEL 20 INDEX ': 'BEL_20_Index',
                                'Euro Stoxx 50 Pr ': 'Euro_Stoxx_50_Pr', 'Austrian Traded Index': 'Austrian_Traded_Index',
                                'IBEX 35': 'IBEX_35', 'PSI 20': 'PSI_20'})

    # ## data1 - Average European Index

    # In[4]:

    data1 = data.copy()
    data1['Average'] = (data1['FTSE_100_Index'] + data1['DAX_Index'] + data1['CAC_40_Index'] + data1['AEX_Index'] + data1['FTSE_Italia'] +
                        data1['BEL_20_Index'] + data1['Euro_Stoxx_50_Pr'] + data1['Austrian_Traded_Index'] + data1['IBEX_35'] + data1['PSI_20'])/10
    data1['Avg_R'] = (data1.Average - data1.Average.shift(1)
                      ) / data1.Average.shift(1)
    data1['Avg_Z'] = (data1['Avg_R'] - (data1['Avg_R'].rolling(252).mean())
                      ) / (data1['Avg_R'].rolling(252).std())
    data1['Avg_o_1d'] = data1['Avg_Z'].apply(
        lambda x: 1 if x < -1.645 or x > 1.645 else 0)
    data1['Avg_o_5d'] = data1['Avg_o_1d'].rolling(5).sum()
    data1['Avg_o_100d'] = data1['Avg_o_1d'].rolling(100).sum()
    data1['Avg_O'] = (data1['Avg_o_1d'] + data1['Avg_o_5d'] +
                      data1['Avg_o_100d']) / 10 - 1
    data1 = data1.set_index('Date')
    Ave = data1[['Avg_O']].dropna()['2003-12-29':]

    # ## Length

    # In[7]:

    def corr_matrix(ret, thresh=0.95, window=250, enddate="2017-01-24", shrinkage=None, index_ret=None, exp_shrinkage_theta=125, detrended=False):
        """Generates correlation matrix for a window that ends on enddate. Correlation can have exponential shrinkage (giving more weights to recent observations.)
        index_ret is used for detrending. If None, will use average return of all assets.
        Will only use assets with more than thresh%% data available in the window"""
        end = list(ret.index).index(enddate) + 1
        start = end - window
        subret = ret.values[start:end]
        if not (index_ret is None):
            end = list(index_ret.index).index(enddate) + 1
            start = end - window
            index_subret = index_ret.values[start:end].flatten()
        eligible = (~np.isnan(subret)).sum(axis=0) >= thresh * window
        subret = subret[:, eligible]
        index_names = ret.columns[eligible]
        # drop whole column when there are less than or equal to
        # thresh number of non-nan entries in the window
        # sub = ret[start:end]
        subret[np.isnan(subret)] = 0
        if detrended:
            r = subret
            if not (index_ret is None):
                I = index_subret
            else:
                I = subret.mean(axis=1)
            n = len(I)
            alpha = (r.sum(axis=0) * (I * I).sum() - I.sum() *
                     r.T.dot(I)) / (n * (I * I).sum() - (I.sum()) ** 2)
            beta = (n * r.T.dot(I) - I.sum() * r.sum(axis=0)) / \
                (n * (I * I).sum() - (I.sum()) ** 2)
            c = r - alpha - np.outer(I, beta)
            # temp = pd.DataFrame(c)
            # temp.index = company_names
            # temp.columns = company_names
            subret = c
        if shrinkage is None:
            corr_mat = pd.DataFrame(np.corrcoef(subret, rowvar=False))
            corr_mat.columns = index_names
            corr_mat.index = index_names
        # elif shrinkage == "LedoitWolf":
        #     cov = ledoit_wolf(subret, assume_centered=True)[0]
        #     std = np.sqrt(np.diagonal(cov))
        #     corr_mat = (cov / std[:, None]).T / std[:, None]
        #     np.fill_diagonal(corr_mat, 1.0)
        #     corr_mat = pd.DataFrame(data=corr_mat, index=subret.columns, columns=subret.columns)
        elif shrinkage == "Exponential":
            stocknames = index_names
            weight_list = np.exp(
                (np.arange(1, window + 1) - window) / exp_shrinkage_theta)
            weight_list = weight_list / weight_list.sum()
            cov = np.cov(subret, rowvar=False, aweights=weight_list)
            cov_diag = np.sqrt(np.diag(cov))
            corr = (cov / cov_diag).T / cov_diag
            corr_mat = pd.DataFrame(corr)
            corr_mat.columns = stocknames
            corr_mat.index = stocknames
        else:
            print("'shrinkage' can only be None or 'Exponential'")
            return None
        # corr_mat.apply(lambda x:1-x**2 if not math.isnan(x) else np.nan)
        return corr_mat

    def all_corr(ret, thresh=0.95, inclusion=pd.DataFrame(), window=250, shrinkage=None, exp_shrinkage_theta=125, detrended=False, store=None):
        """Computes correlations on all dates in the ret dataframe"""
        print("Computing all correlations with window=%s, shrinkage=%s, theta=%s..." % (
            window, shrinkage, exp_shrinkage_theta))
        if store is None:
            allcorr = {}
        else:
            allcorr = store
        alldates = ret.index
        alldates.sort_values()
    #     bar = progressbar.ProgressBar(max_value=len(alldates[window:]))
        for d in alldates[window:]:
            if inclusion.empty:
                allcorr[str(d.strftime("%Y-%m-%d"))] = corr_matrix(ret, thresh, window, enddate=d,
                                                                   shrinkage=shrinkage, exp_shrinkage_theta=exp_shrinkage_theta, detrended=detrended)
            else:
                eligible_stocks = list(inclusion[(inclusion['from'] <= d) & (
                    (inclusion['thru'].isnull()) | (inclusion['thru'] >= d))]['PERMNO'].unique())
                allcorr[str(d.strftime("%Y-%m-%d"))] = corr_matrix(ret[eligible_stocks], thresh, window,
                                                                   enddate=d, shrinkage=shrinkage, exp_shrinkage_theta=exp_shrinkage_theta, detrended=detrended)
    #         bar+=1
        alldates = np.array(sorted([s[-10:] for s in allcorr.keys()]))
        return allcorr

    def build_graph(corr, method='gower'):
        """Builds igraph graph from correlation matrix."""
        if method == "gower":
            def distance(weight):
                return (2 - 2 * weight) ** 0.5  # gower
        elif method == "power":
            def distance(weight):
                return 1 - weight ** 2  # power
        node_names = corr.columns.values
        g = ig.Graph.Weighted_Adjacency(
            corr.values.tolist(), mode="UNDIRECTED", attr="weight", loops=False)
        g.vs['name'] = node_names
        g.es['weight+1'] = np.array(g.es['weight']) + 1.0
        g.es['length'] = distance(np.array(g.es['weight']))
        g.es['absweight'] = np.abs(np.array(g.es['weight']))
        return g

    def MST(corrs, method="gower"):
        """Returns a dictionary of Minimum Spanning Tree for each end date and their graphs in a separate dict"""
        trees = {}
        graphs = {}
        print("Creating MSTs...")
        for d in corrs.keys():
            G = build_graph(corrs[d], method)
            graphs[d[-10:]] = G
            T = G.spanning_tree(return_tree=True, weights='length')
            trees[d[-10:]] = T
        return trees, graphs

    # ## data2 - Correlation

    # In[8]:

    data2 = data.copy()
    data2['Date'] = pd.to_datetime(data2['Date'])
    data2 = data2.set_index('Date', drop=True)
    data2 = data2.astype('float')

    # In[9]:

    all_correlation = all_corr(
        data2, thresh=0.75, shrinkage='Exponential', detrended=False)

    # In[10]:

    trees, graphs = MST(all_correlation, "gower")

    # In[12]:

    length_list = []

    keys = sorted(
        trees.keys(), key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    for i in keys:
        length_list.append(sum(trees[i].es['weight']))

    length = pd.DataFrame(data=length_list, index=keys, columns=['Length'])

    # ## The number of clusters

    # In[13]:

    def construct_clusters(trees, method='Newman', n_of_clusters=None):
        """input: trees: iGraph trees
                method: 'Newman' or 'ClausetNewman'
            Returns dicts of the clusters as lists and igraph.clustering objects"""
        print("Computing clusters...")
        sorteddates = sorted(trees.keys())
        usabletrees = trees
        ig.arpack_options.maxiter = 500000
        clusters = {}
        IGclusters = {}
        print("Computing clusterings using method=%s, n_of_clusters=%s" %
              (method, n_of_clusters))
    #     bar = progressbar.ProgressBar(max_value=len(sorteddates))
        count = 0
        if method == 'Newman':
            for t in sorteddates:
                if len(usabletrees[t].vs) == 0:
                    sys.exit(
                        "There are no stocks with sufficient data on %s! Exiting now." % t)
                if n_of_clusters is None:
                    c = usabletrees[t].community_leading_eigenvector(
                        weights="weight+1")
                else:
                    if len(usabletrees[t].vs) < n_of_clusters:
                        print(
                            "On %s, there are only %s available entities but requiring %s clusters. Use %s clusters instead." % (
                                t, str(len(usabletrees[t].vs)), str(n_of_clusters), str(len(usabletrees[t].vs))))
                    extra = 0
                    length = 0
                    while length != n_of_clusters:
                        c = usabletrees[t].community_leading_eigenvector(weights="weight+1", clusters=min(n_of_clusters,
                                                                                                          len(usabletrees[
                                                                                                              t].vs)) + extra)
                        if len(c) == length:
                            break
                        length = len(c)
                        extra = extra + 1
                IGclusters[t] = c
                clusters[t] = list(c)
                for i in range(0, len(c)):
                    clusters[t][i] = [usabletrees[t].vs["name"][j]
                                      for j in c[i]]
                count = count + 1
    #             bar.update(count)
            return clusters, IGclusters
        elif method == 'ClausetNewman':
            for t in sorteddates:
                if len(usabletrees[t].vs) == 0:
                    sys.exit(
                        "There are no stocks with sufficient data on %s! Exiting now." % t)
                if n_of_clusters is None:
                    c = usabletrees[t].community_fastgreedy(
                        weights="weight+1").as_clustering()
                else:
                    if len(usabletrees[t].vs) < n_of_clusters:
                        print(
                            "On %s, there are only %s available entities but requiring %s clusters. Use %s clusters instead." % (
                                t, str(len(usabletrees[t].vs)), str(n_of_clusters), str(len(usabletrees[t].vs))))
                    c = usabletrees[t].community_fastgreedy(weights="weight+1").as_clustering(
                        n=min(n_of_clusters, len(usabletrees[t].vs)))
                clusters[t] = list(c)
                IGclusters[t] = c
                for i in range(0, len(c)):
                    clusters[t][i] = [usabletrees[t].vs["name"][j]
                                      for j in c[i]]
                count = count + 1
    #             bar.update(count)
            return clusters, IGclusters
        elif method == 'infomap':
            for t in sorteddates:
                if len(usabletrees[t].vs) == 0:
                    sys.exit(
                        "There are no stocks with sufficient data on %s! Exiting now." % t)
                if len(usabletrees[t].vs) < n_of_clusters:
                    print(
                        "On %s, there are only %s available entities but requiring %s clusters. Use %s clusters instead." % (
                            t, str(len(usabletrees[t].vs)), str(n_of_clusters), str(len(usabletrees[t].vs))))
                c = usabletrees[t].community_infomap(edge_weights="weight+1")
                clusters[t] = list(c)
                IGclusters[t] = c
                for i in range(0, len(c)):
                    clusters[t][i] = [usabletrees[t].vs["name"][j]
                                      for j in c[i]]
                count = count + 1
    #             bar.update(count)
            return clusters, IGclusters
        else:
            print(
                "'method' can only be 'Newman' or 'ClausetNewman' or 'infomap'. Your input was '%s'" % method)
            return None

    # In[14]:

    clusters, IGclusters = construct_clusters(trees, method='ClausetNewman')

    # In[15]:

    n_of_clusters_list = []

    for i in keys:
        n_of_clusters_list.append(len(clusters[i]))

    n_of_clusters = pd.DataFrame(
        data=n_of_clusters_list, index=keys, columns=['Number'])

    # ## data3 - Modified/Regime

    # In[19]:

    O_mod = Ave['Avg_O'] + 1/length['Length'].values + \
        n_of_clusters['Number'].values/10

    # In[20]:

    R = ['Low' if O_mod[i] >= -1 and O_mod[i] < 0 else 'Medium' if O_mod[i]
         >= 0 and O_mod[i] < 1 else 'High' for i in O_mod.index]

    # In[21]:

    O_mod = pd.DataFrame(O_mod)
    O_mod['Regime'] = R

    # In[22]:

    data3 = O_mod.copy()

    # ## Volatility

    # In[23]:

    d = data1[['Avg_O']].dropna().copy()
    roller = d.rolling(250)
    volList = roller.std(ddof=0)

    # In[24]:

    volList = volList.dropna()
    volList = volList.rename(columns={'Avg_O': 'Volatility'})

    # ### Graphs

    # In[36]:

    def format_func(value, tick_number):
        return data.iloc[int(value)]['Date'][0:10]

    # ## Fig1, Bokeh Graph - Eurozone market regimes

    # In[37]:

    fig1 = figure(title="Eurozone Market Regimes", plot_width=750, plot_height=400,
                  x_axis_type="datetime", tools='pan,wheel_zoom,box_zoom,reset,hover,save', toolbar_location=None)

    fig1.scatter(data3[data3['Regime'] == 'Low'].index, data3[data3['Regime']
                                                              == 'Low']['Avg_O'].values, size=5, color='green', legend='Low')
    fig1.scatter(data3[data3['Regime'] == 'Medium'].index, data3[data3['Regime']
                                                                 == 'Medium']['Avg_O'].values, size=5, color='blue', legend='Medium')
    fig1.scatter(data3[data3['Regime'] == 'High'].index, data3[data3['Regime']
                                                               == 'High']['Avg_O'].values, size=5, color='red', legend='High')
    fig1.line(volList.index, volList['Volatility'],
              line_color='black', legend='Volatility', line_width=2)

    hover = fig1.select(dict(type=HoverTool))
    hover.tooltips = [("Date", "@x{%F}"),  ("Value", "@y"), ]
    hover.formatters = {'x': 'datetime'}

    # ## FTSE
    # ### Closeness Centrality

    # In[38]:

    keys = sorted(all_correlation.keys(),
                  key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

    centr_closeness = pd.DataFrame(
        index=all_correlation['2014-10-02'].columns.values)

    for d in keys:
        G = build_graph(all_correlation[d], "gower")
        centr_closeness[d] = pd.DataFrame(
            G.closeness(weights='length'), index=G.vs['name'])

    # In[39]:

    centr_closeness = centr_closeness.T
    FTSE_centr_c = centr_closeness[['FTSE_100_Index']].rename(
        columns={'FTSE_100_Index': 'Closeness'})

    # ## Normalize

    # In[41]:

    FTSE_centr_c = FTSE_centr_c['2004-12-30':]
    FTSE_centr_c['Closeness'] = FTSE_centr_c['Closeness'] / \
        FTSE_centr_c['Closeness'].max()

    # ## Betweenness Centrality

    # In[42]:

    centr_betweenness = pd.DataFrame(
        index=all_correlation['2014-10-02'].columns.values)

    for d in keys:
        G = build_graph(all_correlation[d], "gower")
        centr_betweenness[d] = pd.DataFrame(
            G.betweenness(weights='length'), index=G.vs['name'])

    # ## data4

    # In[44]:

    data4 = data.fillna('ffill')
    data4['FTSE100_R'] = (
        data.FTSE_100_Index - data.FTSE_100_Index.shift(1))/data.FTSE_100_Index.shift(1)
    data4['FTSE100_Z'] = (data4['FTSE100_R'] - (data4['FTSE100_R'].rolling(252).mean())) / \
        (data4['FTSE100_R'].rolling(252).std())  # Risk-adjusted Rate of Return
    data4['FTSE100_o_1d'] = data4['FTSE100_Z'].apply(
        lambda x: 1 if x < -1.645 or x > 1.645 else 0)
    data4.set_index('Date', drop=True, inplace=True)

    data4['FTSE100_o_5d'] = data4['FTSE100_o_1d'].rolling(5).sum()
    data4['FTSE100_o_100d'] = data4['FTSE100_o_1d'].rolling(100).sum()
    data4['FTSE100_O'] = (data4['FTSE100_o_1d'] +
                          data4['FTSE100_o_5d'] + data4['FTSE100_o_100d']) / 10 - 1
    FTSE_O = data4[['FTSE100_O']]

    # ## data5 - Index/Regime

    # In[45]:

    FTSE_mod = FTSE_O['FTSE100_O'] + FTSE_centr_c['Closeness']
    R = ['Low' if FTSE_mod[i] >= -1 and FTSE_mod[i] < 0 else 'Medium' if FTSE_mod[i]
         >= 0 and FTSE_mod[i] < 1 else 'High' for i in FTSE_mod.index]

    # In[46]:

    FTSE_mod = pd.DataFrame(FTSE_mod)
    FTSE_mod['Regime'] = R
    data5 = FTSE_mod.rename(columns={0: 'Index'})

    # ## Volatility

    # In[47]:

    d2 = data5[['Index']].dropna().copy()
    roller2 = d2.rolling(250)
    volList2 = roller2.std(ddof=0)
    volList2 = volList2.dropna()
    volList2 = volList2.rename(columns={'Index': 'Volatility'})

    # ## fig2 - US Market Regimes

    # In[49]:

    fig2 = figure(title="UK Market Regimes", plot_width=750, plot_height=400,
                  x_axis_type="datetime", tools='pan,wheel_zoom,box_zoom,reset,hover,save', toolbar_location=None)

    fig2.scatter(data5[data5['Regime'] == 'Low'].index, data5[data5['Regime']
                                                              == 'Low']['Index'].values, size=5, color='green', legend='Low')
    fig2.scatter(data5[data5['Regime'] == 'Medium'].index, data5[data5['Regime']
                                                                 == 'Medium']['Index'].values, size=5, color='blue', legend='Medium')
    fig2.scatter(data5[data5['Regime'] == 'High'].index, data5[data5['Regime']
                                                               == 'High']['Index'].values, size=5, color='red', legend='High')
    fig2.line(volList2.index, volList2['Volatility'],
              line_color='black', legend='Volatility', line_width=2)

    hover = fig2.select(dict(type=HoverTool))
    hover.tooltips = [("Date", "@x{%F}"),  ("Value", "@y"), ]
    hover.formatters = {'x': 'datetime'}


    # ## data_us, US - ETF

    # In[71]:

    data_us = pd.read_excel(
        './dashboard/data/ETF_SSGA.xlsx', sheet_name='Valuefrom 2000', skiprows=5)

    # In[72]:

    data_us = data_us.rename(columns={'Unnamed: 0': 'Date'})
    data_us.dropna()
    data_us['Date'] = pd.to_datetime(data_us['Date'])
    data_us['Date'] = pd.to_datetime(data_us['Date'], format='%Y-%m-%d')
    data_us = data_us[['Date', 'XLB', 'XLE', 'XLF',
                       'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']]

    # ## Average Index

    # In[73]:

    data_us1 = data_us.copy()
    data_us1['Average'] = (data_us1['XLB'] + data_us1['XLE'] + data_us1['XLF'] + data_us1['XLI'] +
                           data_us1['XLK'] + data_us1['XLP'] + data_us1['XLU'] + data_us1['XLV'] + data_us1['XLY'])/9
    data_us1['Avg_R'] = (data_us1.Average -
                         data_us1.Average.shift(1)) / data_us1.Average.shift(1)
    data_us1['Avg_Z'] = (data_us1['Avg_R'] - (data_us1['Avg_R'].rolling(252).mean())
                         ) / (data_us1['Avg_R'].rolling(252).std())
    data_us1['Avg_o_1d'] = data_us1['Avg_Z'].apply(
        lambda x: 1 if x < -1.645 or x > 1.645 else 0)
    data_us1['Avg_o_5d'] = data_us1['Avg_o_1d'].rolling(5).sum()
    data_us1['Avg_o_100d'] = data_us1['Avg_o_1d'].rolling(100).sum()
    data_us1['Avg_O'] = (data_us1['Avg_o_1d'] +
                         data_us1['Avg_o_5d'] + data_us1['Avg_o_100d']) / 10 - 1
    data_us1.dropna(subset=['Date'])
    data_us1 = data_us1.set_index("Date")
    data_us1 = data_us1.dropna()
    Ave_O_us = data_us1[['Avg_O']]
    Ave_O_us.shape

    # ## data_us2, Length, correlation, trees, graphs

    # In[74]:

    data_us2 = data_us.copy()
    data_us2 = data_us.set_index('Date', drop=True)
    data_us2 = data_us2.astype('float')
    data_us2 = data_us2.dropna()

    # In[75]:

    all_correlation = all_corr(
        data_us2, thresh=0.75, shrinkage='Exponential', detrended=False)

    # In[76]:

    trees, graphs = MST(all_correlation, "gower")

    # In[77]:

    length_list = []

    keys = sorted(
        trees.keys(), key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    for i in keys:
        length_list.append(sum(trees[i].es['weight']))

    length = pd.DataFrame(data=length_list, index=keys, columns=['Length'])
    length = length['2000-12-29':]

    # ## The number of clusters

    # In[78]:

    clusters, IGclusters = construct_clusters(trees, method='ClausetNewman')

    # In[79]:

    n_of_clusters_list = []

    for i in keys:
        n_of_clusters_list.append(len(clusters[i]))

    n_of_clusters = pd.DataFrame(
        data=n_of_clusters_list, index=keys, columns=['Number'])
    n_of_clusters = n_of_clusters['2000-12-29':]

    # In[81]:

    O_mod = Ave_O_us['Avg_O'] + 1/length['Length'].values + \
        n_of_clusters['Number'].values/10

    # In[87]:

    R = ['Low' if O_mod[i] >= -1 and O_mod[i] < 0 else 'Medium' if O_mod[i]
         >= 0 and O_mod[i] < 1 else 'High' for i in O_mod.index]

    # In[88]:

    O_mod = pd.DataFrame(O_mod)
    O_mod['Regime'] = R

    # In[89]:

    data6 = O_mod.copy()

    # In[90]:

    d = data_us1[['Avg_O']].dropna().copy()
    roller = d.rolling(250)
    volList3 = roller.std(ddof=0)
    volList3 = volList3.dropna()
    volList3 = volList3.rename(columns={'Avg_O': 'Volatility'})

    # ## fig3

    # In[91]:

    fig3 = figure(title="US Market Regimes", plot_width=750, plot_height=400,
                  x_axis_type="datetime", tools='pan,wheel_zoom,box_zoom,reset,hover,save', toolbar_location=None)

    fig3.scatter(data6[data6['Regime'] == 'Low'].index, data6[data6['Regime']
                                                              == 'Low']['Avg_O'].values, size=5, color='green', legend='Low')
    fig3.scatter(data6[data6['Regime'] == 'Medium'].index, data6[data6['Regime']
                                                                 == 'Medium']['Avg_O'].values, size=5, color='blue', legend='Medium')
    fig3.scatter(data6[data6['Regime'] == 'High'].index, data6[data6['Regime']
                                                               == 'High']['Avg_O'].values, size=5, color='red', legend='High')
    fig3.line(volList3.index, volList3['Volatility'],
              line_color='black', legend='Volatility', line_width=2)

    hover = fig3.select(dict(type=HoverTool))
    hover.tooltips = [("Date", "@x{%F}"),  ("Value", "@y"), ]
    hover.formatters = {'x': 'datetime'}


    l = layout(children = [
            [fig1],
            [fig2],
            [fig3]
            ])

    tab = Panel(child=l, title='Market Indicator')

    return tab



