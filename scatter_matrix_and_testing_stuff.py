# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:07:25 2021

@author: Peter
"""
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from IPython.display import display

#results_data = pd.read_csv("results.csv")
spi_data = os.path.join(".", "spi_global_rankings_intl.csv")
results_data = os.path.join(".", "results.csv")


#data frame of results
df_r = pd.read_csv(results_data, na_values=['NA', '?'])

#data frame of spi values
df_spi = pd.read_csv(spi_data, na_values=['NA', '?'])

#merge to find corelation
df = df_spi.merge(df_r, left_on='name', right_on='home_team')
df = df.drop(columns=['home_team', 'tournament', 'city', 'country', 'neutral', 'date'])
print(df)

#df.to_excel('output.xlsx', sheet_name='teams, spi & scores')

#df = df.select_dtypes(include=['int', 'float'])
#print(df)

#scatter matrix
#pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(15, 10))

#seperate in test and target
X_all = df.drop(columns=['home_score', 'away_score'])
#y_all = df.drop(columns=['rank', 'name', 'confed', 'off', 'def', 'spi', 'away_team'])
y_all = df['home_score']
print(y_all)

#make all values small
from sklearn.preprocessing import scale

cols = [['off', 'def', 'spi']]
for col in cols:
    X_all[col] = scale(X_all[col])
    
print(X_all)

def preprocess_feature(X):
    output = pd.DataFrame(index = X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
        output = output.join(col_data)
    return output

X_all = preprocess_feature(X_all)
#print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

#shuffle and split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=50, random_state=71)

from time import time

def train_classifier(classifier, X_train, y_train):
    #time for comparisson
    start_time = time()
    classifier.fit(X_train, y_train)
    end_time = time()
    
    print("Model trained in: {:.4f}".format(end_time - start_time))
    
from sklearn.metrics import f1_score
    
def predict_labels(classifier, features, target):
    start_time = time()
    y_pred = classifier.predict(features)
    end_time = time()
    
    print("Prediction took: {:.4f}".format(end_time - start_time))
    
    return f1_score(target, y_pred), sum(target == y_pred) / float(len(y_pred))

def train_predict(classifier, X_train, y_train, X_test, y_test):
    print("Training {} using a training size of {} . . .".format(classifier.__class__.__name__, len(X_train)))
    
    train_classifier(classifier, X_train, y_train)
    
    f1, accuracy = predict_labels(classifier, X_train, y_train)
    print(f1, accuracy)
    print("f1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, accuracy))

    f1, accuracy = predict_labels(classifier, X_test, y_test)
    print("f1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , accuracy))    
    
from sklearn.svm import SVC

model = SVC(kernel='linear', C=1E10)
model.fit(X_all, y_all)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, s=50, cmap='autumn')
plot_svc_decision_function(model)

#classifier_svc = SVC(random_state=54, kernel='rbf')
#train_predict(classifier_svc, X_train, y_train, X_test, y_test)