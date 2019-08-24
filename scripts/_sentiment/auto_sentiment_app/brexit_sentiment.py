#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ericyuan
"""
import dill
import joblib
import pandas as pd
from textblob import TextBlob

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

class Textblob:
    def __mapfun(self, text):
        '''
        text: string
        '''
        text = TextBlob(text)
        result = text.sentiment
        return (result[0], result[1])
    
    def fit(self, textSeries):
        '''
        Input: list/tuple/pandas.Series [sentence, sentence, ...]
        Output: dataframe with column name "polar", "subj"
        
        Explanation of output:
            (1) The polarity score is a float within the range [-1.0, 1.0]
            where -1.0 is very negative and 1.0 is very positive
            (2) The subjectivity is a float within the range [0.0, 1.0] 
            where 0.0 is very objective and 1.0 is very subjective.
        '''
        txt = pd.Series(textSeries)
        txt = txt.map(self.__mapfun)
        txt = txt.apply(pd.Series)
        txt.columns = ['polar', 'subj']
        return txt
    
# sentiment classifier (trained with tweets data)
twitter_tfidf = dill.load(open("models/stwitter_tfidf.pickle", 'rb'))
twitter_logit = joblib.load("models/stwitter_logit.pickle")

# merge data
brexit1 = pd.read_csv("../../result/brexitText_cleaned.csv")
del brexit1['Unnamed: 0']
del brexit1['id']
brexit2 = pd.read_csv("../../result/cleaned_newbrexit.csv")
del brexit2['Unnamed: 0']
allbrexit = pd.concat([brexit1, brexit2])
allbrexit.index = [i for i in range(len(allbrexit))]
allbrexit = allbrexit.dropna()

# result
result = allbrexit[['time']]

# Sentiment analysis
Textblob_sentiment = Textblob()
# body sentiment (tweets classifier)
logit_body_sentiment = twitter_logit.predict(twitter_tfidf.transform(allbrexit.body))
# body sentiment (Textblob)
textblob_body_sentiment = Textblob_sentiment.fit(allbrexit.body)
textblob_body_sentiment.columns = ['body_polar', 'body_subj']
# headline sentiment (tweets classifier)
logit_headline_sentiment = twitter_logit.predict(twitter_tfidf.transform(allbrexit.headline))
# headline sentiment (Textblob)
textblob_headline_sentiment = Textblob_sentiment.fit(allbrexit.headline)
textblob_headline_sentiment.columns = ['headline_polar', 'headline_subj']

result['body_sentiment_logit'] = logit_body_sentiment
result['headline_sentiment_logit'] = logit_headline_sentiment
result = pd.concat([result, textblob_body_sentiment, textblob_headline_sentiment], axis=1, join='inner')
result.index = [i for i in range(len(result))]
result.to_csv("../../result/allBrexit_sentiment.csv")






