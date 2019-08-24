# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from sklearn.externals import joblib
import dill

import sentiment_pipeline

# read data
tweets = pd.read_csv("../../data/clean_tweet.csv")
del tweets['Unnamed: 0']

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# ------------------------------------------------------------------
# --------------------------- INPUT PART ---------------------------
# ------------------------------------------------------------------
# Data Input
def simple_clean(wordlist):
    if len(wordlist) <= 2:
        return np.nan
    else:
        return list(filter(lambda x: len(x) > 3, wordlist))
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts, stop_words = stop_words):
    return [w for w in texts if not w in stop_words]
body = tweets['text'].drop_duplicates().dropna()
body = pd.Series(sent_to_words(body))
body = body.map(remove_stopwords).map(simple_clean).dropna()
sents = body.map(lambda x: ' '.join(x))
sents = pd.concat([sents.head(500), sents.tail(500)])

# ------------------------------------------------------------------
# ----------------------- Sentiment Features -----------------------
# ------------------------------------------------------------------
FEATURE = sentiment_pipeline.Sentiment_feature()
phrase_model, tfidf_matrix, fe_names = FEATURE.gensim_phrase(documents = sents, \
                                                             total_features = 10000)
# save tfidf model
dill.dump(phrase_model, open("models/tweet_tfidf.pickle", "wb"))

# ------------------------------------------------------------------
# ------------------------ Sentiment models ------------------------
# ------------------------------------------------------------------
y = pd.concat([tweets['target'][body.index].head(500), \
               tweets['target'][body.index].tail(500)])

TRAIN = sentiment_pipeline.Sentiment_train()
TRAIN.randomforest(tfidf_matrix, y, \
                   filepath = "models/sentiment_randomforest.sav")

TRAIN.logisticReg(tfidf_matrix, y,\
                   filepath = "models/sentiment_logistic.sav")

# ------------------------------------------------------------------
# --------------------------- Deployment ---------------------------
# ------------------------------------------------------------------
phrase_model = dill.load(open("models/tweet_tfidf.pickle", 'rb'))
loaded_rf = joblib.load("models/sentiment_randomforest.sav")
loaded_logistic = joblib.load("models/sentiment_logistic.sav")

loaded_logistic.predict(phrase_model.transform(['suashuah']))


# ------------------------------------------------------------------
# -------------------------- Shadi Logit ---------------------------
# ------------------------------------------------------------------
# sentiment_pipeline.shadi_logit()
# Accuracy: 82.23%
shadi_tfidf = dill.load(open("models/shadi_tfidf.pickle", 'rb'))
sagdi_logit = joblib.load("models/shadi_logit.pickle")
sagdi_logit.predict(shadi_tfidf.transform(['good great yes']))


