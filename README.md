# Dashborad
#### Visualizing a research project on predicting the GBPUSD rate, supervised by [Eugene Neduv](https://datascience.columbia.edu/eugene-neduv)
- Features are *sentiment scores* and volumes in news articles, and option-implied volatility.
  - *sentiment scores* is defined as a normalized number of news sentiment.
  - We gathered all news with *Brexit* tags from Reuter news archive from 2015 to 2018.
- We take those three features and express them as a combined probability function
#### Used [Bokeh](https://bokeh.pydata.org/en/latest/index.html) and ran a Bokeh server on my laptop
#### My responsibilities include
- writing this dashboard from the scratch.
- consolidating and verifying existing works.


## Screenshots

### Two Plots on Sentiment Analyses
#### 1)Sub-topic: F/X Market
- **Aggregation frequency**: Weekly
- **Aggregation method**: Proportion of positive news
- This sub-topic was manually identified after applying *topic modeling*

![FX Market](screenshots/Sentiment_FX_market_positives.png)

#### 2)Sub-topic: Global Financial Market
- **Aggregation frequency**: Weekly
- **Aggregation method**: Proportion of positive news

![Stock Market](screenshots/Sentiment_financial_markets_positives.png)

### A Plot on Combined Probability
- Three features are plotted at the bottom
  - Option-implied volatility, sentiment scores (polarity), news volumes in order (left to right)

![Combined probability](screenshots/combined_probability.png)
