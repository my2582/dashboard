#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ericyuan
"""
# features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases, Phraser
# model
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
# evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
# save and load model
from sklearn.externals import joblib
# plot
import matplotlib.pyplot as plt
import dill
import pandas as pd

class Sentiemnt_input:
    def __init__(self, data, feature, label):
        '''
        data: dataframe
        '''
        self.data = data
        self.text = data[feature]
        self.feature = feature
        self.label = label
        print("Loading data...")
        print("Length of dataset: {0}".format(len(data)))
        
    def cleaning(self, cleanfunc):
        self.data[self.feature] = self.text.map(cleanfunc)
        return self.data

class Sentiment_feature:
    # document: Series
    def count_vectorizer(self, documents, total_features, \
                         param = {"stop_words": 'english'}):
        #  Count Vectorizer
        count_vectorizer = CountVectorizer(max_features=total_features, **param)
        count = count_vectorizer.fit_transform(documents)
        count_feature_names = count_vectorizer.get_feature_names()
        return count_vectorizer, count, count_feature_names
    
    def tfidf_vectorizer(self, documents, total_features, \
                         param = {"stop_words": 'english'}):
        #  TFIDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=total_features, **param)
        tfidf = tfidf_vectorizer.fit_transform(documents)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        return tfidf_vectorizer, tfidf, tfidf_feature_names
    
    def gensim_phrase(self, documents, total_features, \
                      param_phrase = {"progress_per": 10000},\
                      param_tfidf = {"stop_words": 'english',\
                                     "analyzer": "word",\
                                     "tokenizer": lambda x: x,\
                                     "preprocessor": lambda x: x}):
        sent = documents.map(lambda x: x.split())
        phrases = Phrases(sent, **param_phrase)
        bigram = Phraser(phrases)
        sentences = bigram[sent]
        tfidf_model, tfidf, tfidf_feature_names = self.tfidf_vectorizer(sentences, 1000, \
                                                           param = param_tfidf)
        return tfidf_model, tfidf, tfidf_feature_names

class Sentiment_train:
    def splitData(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def logisticReg(self, X, y, param = {"solver": "liblinear"}, \
                    filepath = "models/sentiment_logistic.sav"):
        X_train, X_test, y_train, y_test = self.splitData(X, y)
        # train the model
        logisticReg = linear_model.LogisticRegression(**param)
        logisticReg.fit(X_train, y_train)
        pred_trainy = logisticReg.predict(X_train)
        pred_testy = logisticReg.predict(X_test)
        # evaluation
        trainScore = accuracy_score(y_train, pred_trainy)
        testScore = accuracy_score(y_test, pred_testy)
        # print score
        print("Logistic regression accuracy on train dataset: {0}".format(round(trainScore, 4)))
        print("Logistic regression accuracy on test dataset: {0}".format(round(testScore, 4)))
        # save model
        print("save model named sentiment_logistic")
        print()
        joblib.dump(logisticReg, filepath)
        
    def naivebayes(self, X, y, param = {"var_smoothing": 1e-9}, \
                   filepath = "models/sentiment_bayes.sav"):
        X_train, X_test, y_train, y_test = self.splitData(X, y)
        # train the model
        bayes = GaussianNB(**param)
        bayes.fit(X_train, y_train)
        pred_trainy = bayes.predict(X_train)
        pred_testy = bayes.predict(X_test)
        # evaluation
        trainScore = accuracy_score(y_train, pred_trainy)
        testScore = accuracy_score(y_test, pred_testy)
        # print score
        print("Naive Bayes accuracy on train dataset: {0}".format(round(trainScore, 4)))
        print("Naive Bayes accuracy on test dataset: {0}".format(round(testScore, 4)))
        # save model
        print("save model named sentiment_bayes")
        print()
        joblib.dump(bayes, filepath)
        
    def randomforest(self, X, y, param = {"n_estimators": 100}, \
                     filepath = "models/sentiment_randomforest.sav"):
        X_train, X_test, y_train, y_test = self.splitData(X, y)
        # train the model
        rf = RandomForestClassifier(**param)
        rf.fit(X_train, y_train)
        pred_trainy = rf.predict(X_train)
        pred_testy = rf.predict(X_test)
        # evaluation
        trainScore = accuracy_score(y_train, pred_trainy)
        testScore = accuracy_score(y_test, pred_testy)
        # print score
        print("Random Forest accuracy on train dataset: {0}".format(round(trainScore, 4)))
        print("Random Forest accuracy on test dataset: {0}".format(round(testScore, 4)))
        # save model
        print("save model named sentiment_randomforest")
        print()
        joblib.dump(rf, filepath)
        
    def gbdt(self, X, y, param = {"n_estimators": 100}, \
             filepath = "models/sentiment_gbdt.sav"):
        X_train, X_test, y_train, y_test = self.splitData(X, y)
        # train the model
        gbdt = GradientBoostingClassifier(**param)
        gbdt.fit(X_train, y_train)
        pred_trainy = gbdt.predict(X_train)
        pred_testy = gbdt.predict(X_test)
        # evaluation
        trainScore = accuracy_score(y_train, pred_trainy)
        testScore = accuracy_score(y_test, pred_testy)
        # print score
        print("Random Forest accuracy on train dataset: {0}".format(round(trainScore, 4)))
        print("Random Forest accuracy on test dataset: {0}".format(round(testScore, 4)))
        # save model
        print("save model named sentiment_gbdt")
        print()
        joblib.dump(gbdt, filepath)

class Sentiment_deploy:
    def predict(self, X, filepath):
        print("Using model from {0}".format(filepath))
        loaded_model = joblib.load(filepath)
        result = loaded_model.predict(X)
        return result
    def bagging(self, X, modelpath_list):
        result = [0 for i in range(len(X))]
        for each_model in modelpath_list:
            loaded_model = joblib.load(each_model)
            result += loaded_model.predict(X)
        return result/len(modelpath_list)
    def stacking(self, X, modelpath_list, metaLearner):
        pass

class ROC_curve:
    def plot(y_pred, y_true):
        fpr, tpr, thresholds = roc_curve(y_pred, y_true)
        roc_auc = auc(fpr, tpr)
        # plot
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

class shadi_logit:
    def __init__(self): 
        # read data
        all_tweets = pd.read_csv("../../data/clean_tweet.csv")
        del all_tweets['Unnamed: 0']
        X_train, X_test, y_train, y_test = train_test_split(all_tweets['text'], \
                                                            all_tweets['target'], \
                                                            random_state=0)
        # tf-idf model
        vectorizer = TfidfVectorizer(max_features = 100000, ngram_range=(1,2))
        vect = vectorizer.fit(X_train)
        dill.dump(vectorizer, open("models/shadi_tfidf.pickle", "wb"))
        # transform
        x_train_vectorized = vect.transform(X_train)
        # ml
        logit_model = LogisticRegression()
        logit_model.fit(x_train_vectorized, y_train)
        dill.dump(logit_model, open("models/shadi_logit.pickle", "wb"))
        # accuracy
        y_pred_2 = logit_model.predict(vectorizer.transform(X_test))
        print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred_2) * 100))
        
    
    
    
    
    
    
    
    