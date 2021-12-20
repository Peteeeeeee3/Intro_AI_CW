import os
#data preprocessing
import pandas as pd
import numpy as np
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
#import xgboost as xgb
#the outcome (dependent variable) has only a limited number of possible values. 
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#A random forest is a meta estimator that fits a number of decision tree classifiers 
#on various sub-samples of the dataset and use averaging to improve the predictive 
#accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier
#a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
#displayd data
from IPython.display import display

#data = pd.read_excel("merge_maestro.xlsx")
data = pd.read_excel('merging_test.xlsx')
data = data.drop(columns=['Unnamed: 0'])
#data_for_matrix = data.drop(columns=['winner'])
print(data)
pd.plotting.scatter_matrix(data, alpha=0.2, figsize=(10, 10))
data = data.drop(data.index[300 : 19385])

print(data)

# Preview data.
#display(data.head())

# Visualising distribution of data
from pandas.plotting import scatter_matrix

#the scatter matrix is plotting each of the columns specified against each other column.
#You would have observed that the diagonal graph is defined as a histogram, which means that in the 
#section of the plot matrix where the variable is against itself, a histogram is plotted.

#Scatter plots show how much one variable is affected by another. 
#The relationship between two variables is called their correlation
#negative vs positive correlation

#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

#scatter_matrix(data[['rank','name','off','def','spi','home_score','away_score', 'winner']], figsize=(15,15))

# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = data.drop(columns=['home_score', 'away_score'])
y_hs_all = data['home_score']
y_as_all = data['away_score']

# Standardising the data.
from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
#, 'home_score', 'away_score'
cols = [['home_off','home_def','home_spi', 'away_off', 'away_def', 'away_spi']]
for col in cols:
    X_all[col] = scale(X_all[col])

#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

print("\nFeature values:")
display(X_all.head())

from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set. (home score)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_hs_all, 
                                                    test_size = 0.2,
                                                    random_state = 15)

#for measuring training time
from time import time 
# F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
#It considers both the precision p and the recall r of the test to compute 
#the score: p is the number of correct positive results divided by the number of 
#all positive results, and r is the number of correct positive results divided by 
#the number of positive results that should have been returned. The F1 score can be 
#interpreted as a weighted average of the precision and recall, where an F1 score 
#reaches its best value at 1 and worst at 0.
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

def predict_match(clf, match):
    y_pred = clf.predict(match)
    return y_pred
    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, pos_label=1, average='micro'), sum(target == y_pred) / float(len(y_pred)), y_pred


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc, y_pred = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc, y_pred = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    return y_pred
    
# Initialize the three models (XGBoost is initialized later)
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

lg = LogisticRegression(random_state = 42)
svc = SVC(random_state = 912, kernel='rbf')
dtc = DecisionTreeClassifier()
kmeans = KMeans(n_clusters=2)

#Boosting refers to this general problem of producing a very accurate prediction rule 
#by combining rough and moderately inaccurate rules-of-thumb
#clf_C = xgb.XGBClassifier(seed = 82)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#-----------------------------------------------------------------------------------------------------------------------
#   away sore
#-----------------------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_all, y_as_all, 
                                                    test_size = 100,
                                                    random_state = 15)


#logistic regression
y_pred_lg = train_predict(lg, X_train, y_train, X_test, y_test)
print('')
print('LG')
print(pd.DataFrame(data=np.c_[y_test, y_pred_lg]))
print('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_lg)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#SVC
y_pred_svc = train_predict(svc, X_train, y_train, X_test, y_test)
print('')
print('SVC')
print(pd.DataFrame(data=np.c_[y_test, y_pred_svc]))
print ('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_svc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#dtc
y_pred_dtc = train_predict(dtc, X_train, y_train, X_test, y_test)
print('')
print('DTC')
print(pd.DataFrame(data=np.c_[y_test, y_pred_dtc]))
print ('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_dtc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#k-means
y_pred_km = train_predict(kmeans, X_train, y_train, X_test, y_test)
print('')
print('KMEANS')
print(pd.DataFrame(data=np.c_[y_test, y_pred_km]))
print('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_km)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#-----------------------------------------------------------------------------------------------------------------------
#   away sore
#-----------------------------------------------------------------------------------------------------------------------

#logistic regression
y_pred_lg = train_predict(lg, X_train, y_train, X_test, y_test)
print('')
print('LG')
print(pd.DataFrame(data=np.c_[y_test, y_pred_lg]))
print('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_lg)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#SVC
y_pred_svc = train_predict(svc, X_train, y_train, X_test, y_test)
print('')
print('SVC')
print(pd.DataFrame(data=np.c_[y_test, y_pred_svc]))
print ('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_svc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#dtc
y_pred_dtc = train_predict(dtc, X_train, y_train, X_test, y_test)
print('')
print('DTC')
print(pd.DataFrame(data=np.c_[y_test, y_pred_dtc]))
print ('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_dtc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#k-means
y_pred_km = train_predict(kmeans, X_train, y_train, X_test, y_test)
print('')
print('KMEANS')
print(pd.DataFrame(data=np.c_[y_test, y_pred_km]))
print('')
#confusion matrix
cm = confusion_matrix(y_test, y_pred_km)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5], title='Normalized confusion matrix')
plt.show()
print('')

#UEFA PREDICTIONS


data_spi = pd.read_csv("spi_global_rankings_intl.csv")
uefa_matches = pd.read_csv("UEFA_matches.csv")

branch_1 = uefa_matches.drop([3, 4, 5, 6, 7, 8])
print(branch_1)
print('')

branch_1 = branch_1.merge(data_spi, left_on='home_team', right_on='name')
print(branch_1)

branch_1 = branch_1.merge(data_spi, left_on='away_team', right_on='name')
print(branch_1)

branch_1 = branch_1.drop(columns=['confed_x', 'confed_y'])

cols = [['off_x','def_x','spi_x', 'off_y', 'def_y', 'spi_y']]
for col in cols:
    branch_1[col] = scale(branch_1[col])

print('')
print("dtc")
X_pred = preprocess_features(branch_1)
#match_pred_dtc = predict_match(dtc, X_pred)
#print(match_pred_dtc)
print('')
print("svc")
#match_pred_svc = predict_match(svc, X_pred)
#print(match_pred_svc)
print('')
print("lg")
match_pred_lg = predict_match(lg, X_pred)
print(match_pred_lg)
print('')
print("km")
match_pred_km = predict_match(kmeans, X_pred)
print(match_pred_km)


from sklearn.metrics import accuracy_score

# # Use 5-fold split
# kf = KFold(5,shuffle=True)

# fold = 1
# # The data is split five ways, for each fold, the 
# # Perceptron is trained, tested and evaluated for accuracy
# for train_index, validate_index in kf.split(X_test,y_test):
#     dtc.fit(X_test[train_index],y_test[train_index])
#     y_test = y_test[validate_index]
#     y_pred = dtc.predict(X_test[validate_index])
#     #print(y_test)
#     #print(y_pred)
#     #print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
#     print(f"Fold #{fold}, Training Size: {len(X_test[train_index])}, Validation Size: {len(X_test[validate_index])}")
#     print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     fold += 1