import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from IPython.display import display
from sklearn.preprocessing import scale
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from time import time 
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#change to the number of iterations of predictions you want
#higher values will lead to better statistical analysis about results
#WARNING: can take very long. 1000 iterations took 30+ hours on my laptop 
iterations = 1

#matches dataset merged with spi-ratings
data = pd.read_excel('merging_test.xlsx')
data = data.drop(columns=['Unnamed: 0'])

#used to understand the effects of all data values on each other
pd.plotting.scatter_matrix(data, alpha=0.2, figsize=(10, 10))

#create targets for both away and home scores
X_all = data.drop(columns=['home_score', 'away_score'])
y_hs_all = data['home_score']
y_as_all = data['away_score']

#Center to the mean and component wise scale to unit variance to increase prediction accuracy
cols = [['home_off','home_def','home_spi', 'away_off', 'away_def', 'away_spi', 'home_rank', 'away_rank']]
for col in cols:
    X_all[col] = scale(X_all[col])
    
X_all = X_all.select_dtypes(include=['int', 'float'])
    
#returns desired score
def predict_match(clf, match):
    y_pred = clf.predict(match)
    return y_pred
    
#predicts labels and returns f1 score
def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
    
    return f1_score(target, y_pred, pos_label=1, average='micro')

#trains classifier and makes a prediction
def train_predict(clf, X_train, y_train, X_test, y_test):
    
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    #train
    clf.fit(X_train, y_train)
    
    #predict    
    f1, acc, y_pred = predict_labels(clf, X_train, y_train)
    
    #show results
    print("F1-score for training set: {:.4f} , {:.4f}.".format(f1))
    
    f1, acc, y_pred = predict_labels(clf, X_test, y_test)
    print("F1 score for test set: {:.4f} , {:.4f}.".format(f1))
    
    return y_pred

#home classifiers
h_lg = LogisticRegression(random_state = 42)
h_svc = SVC(random_state = 912, kernel='rbf')
h_dtc = DecisionTreeClassifier()
h_kmeans = KMeans(n_clusters=2)
#away classifiers
a_lg = LogisticRegression(random_state = 42)
a_svc = SVC(random_state = 912, kernel='rbf')
a_dtc = DecisionTreeClassifier()
a_kmeans = KMeans(n_clusters=2)


#plats a confustion matrix
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
#   home score
#-----------------------------------------------------------------------------------------------------------------------
#the classifiers are trained to create the confusion matrices and to evaluate performance

#Split into train and test data (home score)
h_X_train, h_X_test, h_y_train, h_y_test = train_test_split(X_all, y_hs_all, 
                                                                test_size = 100,
                                                                random_state = 15)
    
    
#logistic regression
y_pred_lg = train_predict(h_lg, h_X_train, h_y_train, h_X_test, h_y_test)
print('')
print('LG')
print(pd.DataFrame(data=np.c_[h_y_test, y_pred_lg]))
print('')
#confusion matrix
cm = confusion_matrix(h_y_test, y_pred_lg)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (Logistic Regression)')
print(cm_normalized)
    
plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (Logistic Regression)')
plt.show()
print('')
    
# #SVC
y_pred_svc = train_predict(h_svc, h_X_train, h_y_train, h_X_test, h_y_test)
print('')
print('SVC')
print(pd.DataFrame(data=np.c_[h_y_test, y_pred_svc]))
print ('')
#confusion matrix
cm = confusion_matrix(h_y_test, y_pred_svc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (Support Vector Machine)')
print(cm_normalized)
    
plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (Support Vector Machine)')
plt.show()
print('')
    
#dtc
y_pred_dtc = train_predict(h_dtc, h_X_train, h_y_train, h_X_test, h_y_test)
print('')
print('DTC')
print(pd.DataFrame(data=np.c_[h_y_test, y_pred_dtc]))
print ('')
#confusion matrix
cm = confusion_matrix(h_y_test, y_pred_dtc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (Decision Tree Classifier)')
print(cm_normalized)
    
plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (Decision Tree Classifier)')
plt.show()
print('')
    
#k-means
y_pred_km = train_predict(h_kmeans, h_X_train, h_y_train, h_X_test, h_y_test)
print('')
print('KMEANS')
print(pd.DataFrame(data=np.c_[h_y_test, y_pred_km]))
print('')
#confusion matrix
cm = confusion_matrix(h_y_test, y_pred_km)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (K-Means)')
print(cm_normalized)
    
plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (K-Means)')
plt.show()
print('')
    
#-----------------------------------------------------------------------------------------------------------------------
#   away score
#-----------------------------------------------------------------------------------------------------------------------
#the classifiers are trained to create the confusion matrices and to evaluate performance


#Split into train and test data (away score)
a_X_train, a_X_test, a_y_train, a_y_test = train_test_split(X_all, y_as_all, 
                                                                test_size = 100,
                                                                random_state = 15)
    
#logistic regression
y_pred_lg = train_predict(a_lg, a_X_train, a_y_train, a_X_test, a_y_test)
print('')
print('LG')
print(pd.DataFrame(data=np.c_[a_y_test, y_pred_lg]))
print('')
#confusion matrix
cm = confusion_matrix(a_y_test, y_pred_lg)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (Logistic Regression)')
print(cm_normalized)
    
plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (Logistic Regression)')
plt.show()
print('')
    
#SVC
y_pred_svc = train_predict(a_svc, a_X_train, a_y_train, a_X_test, a_y_test)
print('')
print('SVC')
print(pd.DataFrame(data=np.c_[a_y_test, y_pred_svc]))
print ('')
#confusion matrix
cm = confusion_matrix(a_y_test, y_pred_svc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (Support Vector Machine)')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (Support Vector Machine)')
plt.show()
print('')
    
#dtc
y_pred_dtc = train_predict(a_dtc, a_X_train, a_y_train, a_X_test, a_y_test)
print('')
print('DTC')
print(pd.DataFrame(data=np.c_[a_y_test, y_pred_dtc]))
print ('')
#confusion matrix
cm = confusion_matrix(a_y_test, y_pred_dtc)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (Decision Tree Classifier)')
print(cm_normalized)
    
plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (Decision Tree Classifier)')
plt.show()
print('')
    
#k-means
y_pred_km = train_predict(a_kmeans, a_X_train, a_y_train, a_X_test, a_y_test)
print('')
print('KMEANS')
print(pd.DataFrame(data=np.c_[a_y_test, y_pred_km]))
print('')
#confusion matrix
cm = confusion_matrix(a_y_test, y_pred_km)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix (K-Means)')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, [0,1,2,3,4,5,6,7,8,9,10], title='Normalized confusion matrix (K-Means)')
plt.show()
print('')

#-----------------------------------------------------------------------------------------------------------------------
#UEFA PREDICTIONS
#-----------------------------------------------------------------------------------------------------------------------

def predict_uefa(home_clf, away_clf, name):
    #this will store the qualified teams and be the return value
    winners = []
    uefa_matches = pd.read_csv("UEFA_matches.csv")
    
    #split matches into the qualification branches and select only the matches with assigned teams
    #there are three branches to qualify from, only teams on the same branch can play against each other
    branch_1 = uefa_matches.drop([2, 3, 4, 5, 6, 7, 8])
    branch_2 = uefa_matches.drop([0, 1, 2, 5, 6, 7, 8])
    branch_3 = uefa_matches.drop([0, 1, 2, 3, 4, 5, 8])

    branches = [branch_1, branch_2, branch_3]

    for branch in branches:
        #remove any data that is not required
        branch = branch.select_dtypes(include=['int', 'float'])
        branch = branch.drop(columns=['match', 'home_score', 'away_score'])
        
        #scale data to improve predictions
        cols = [['home_off','home_def','home_spi', 'away_off', 'away_def', 'away_spi']]
        for col in cols:
            branch[col] = scale(branch[col])
    
    #indicate classifier used
    print('')
    print(name)
    
    #branch 1
    home_scores = []
    away_scores = []
    
    #predict scores of all matches with predefined teams
    for branch in branches:
        home_scores.append(predict_match(home_clf, branch_1))
        away_scores.append(predict_match(away_clf, branch_1))
    
    #reformat the way scores are stored so home and away scores are store as a unit, so as a match so to say
    scores = []
    for i in range(len(home_scores)):
        scores.append([home_scores[i], away_scores[i]])
        
    #finals 1
    final_1 = pd.DataFrame()
    final_1 = uefa_matches.drop([1,2,3,4,5,6,7,8])
    
    #show prediction outcomes of frist two matches of branch
    home_team = uefa_matches['home_team']
    away_team = uefa_matches['away_team']
    print('')
    print('{} {} : {} {}'.format(home_team[0], scores[0][0][0], scores[0][0][1], away_team[0]))
    print('')
    print('{} {} : {} {}'.format(home_team[1], scores[0][1][0], scores[0][1][1], away_team[1]))
    print('')
    
    #add correct home team to finals
    if (scores[0][0] > scores[0][0][1]).any():
        final_1['away_team'] = final_1['away_team'].replace(['Austria'], ' ')
        final_1['away_off'] = final_1['away_off'].replace(['2.1'], '0')
        final_1['away_def'] = final_1['away_def'].replace(['0.79'], '0')
        final_1['away_spi'] = final_1['away_spi'].replace(['73.08'], '0')
        final_1['away_rank'] = final_1['away_rank'].replace(['26'], '0')
    else:
        final_1['home_team'] = final_1['home_team'].replace(['Wales'], 'Austria')
        final_1['home_off'] = final_1['home_off'].replace(['1.78'], '2.1')
        final_1['home_def'] = final_1['home_def'].replace(['0.71'], '0.79')
        final_1['home_spi'] = final_1['home_spi'].replace(['69.51'], '73.08')
        final_1['home_rank'] = final_1['home_rank'].replace(['34'], '26')
        final_1['away_team'] = final_1['away_team'].replace(['Austria'], ' ')
        final_1['away_off'] = final_1['away_off'].replace(['2.1'], '0')
        final_1['away_def'] = final_1['away_def'].replace(['0.79'], '0')
        final_1['away_spi'] = final_1['away_spi'].replace(['73.08'], '0')
        final_1['away_rank'] = final_1['away_rank'].replace(['26'], '0')
        
    #add correct away team to finals
    if (scores[0][1] > scores[0][1][1]).any():
        final_1['away_team'] = final_1['away_team'].replace([' '], 'Scotland')
        final_1['away_off'] = final_1['away_off'].replace(['0'], '1.7')
        final_1['away_def'] = final_1['away_def'].replace(['0'], '0.74')
        final_1['away_spi'] = final_1['away_spi'].replace(['0'], '67.22')
        final_1['away_rank'] = final_1['away_rank'].replace(['0'], '39')
    else:
        final_1['away_team'] = final_1['away_team'].replace([' '], 'Ukraine')
        final_1['away_off'] = final_1['away_off'].replace(['0'], '1.81')
        final_1['away_def'] = final_1['away_def'].replace(['0'], '0.86')
        final_1['away_spi'] = final_1['away_spi'].replace(['0'], '66.47')
        final_1['away_rank'] = final_1['away_rank'].replace(['0'], '42')
        
    #remove data not required
    final_1_sin = final_1.select_dtypes(include=['int', 'float'])
    final_1_sin = final_1_sin.drop(columns=['match', 'home_score', 'away_score'])
    
    #predict score
    finals_score = []
    finals_score.append(predict_match(home_clf, final_1_sin))
    finals_score.append(predict_match(away_clf, final_1_sin))
    
    #show score
    home_team = final_1['home_team']
    away_team = final_1['away_team']
    print('')
    print('{} {} : {} {}'.format(home_team[0], finals_score[0][0], finals_score[1][0], away_team[0]))
    print('')

    #show qualifying team
    if (finals_score > finals_score[1]).any():
        print('{} has qualified!'.format(home_team[0]))
        winners.append(home_team[0])
    else:
        print('{} has qualified!'.format(away_team[0]))
        winners.append(away_team[0])
    print('')
        
    #finals 2
    final_2 = pd.DataFrame()
    final_2 = uefa_matches.drop([0,1,2,4,5,6,7,8])
    
    #show outcome of first two matches on branch
    home_team = uefa_matches['home_team']
    away_team = uefa_matches['away_team']
    print('')
    print('{} {} : {} {}'.format(home_team[3], scores[1][0][0], scores[1][0][1], away_team[3]))
    print('')
    print('{} {} : {} {}'.format(home_team[4], scores[1][1][0], scores[1][1][1], away_team[4]))
    print('')
    
    #add correct home team to finals
    if (scores[1][0] > scores[1][0][1]).any():
        final_2['away_team'] = final_2['away_team'].replace(['Poland'], ' ')
        final_2['away_off'] = final_2['away_off'].replace(['2.06'], '0')
        final_2['away_def'] = final_2['away_def'].replace(['0.81'], '0')
        final_2['away_spi'] = final_2['away_spi'].replace(['72'], '0')
        final_2['away_rank'] = final_2['away_rank'].replace(['28'], '0')
    else:
        final_2['home_team'] = final_2['home_team'].replace(['Russia'], 'Poland')
        final_2['home_off'] = final_2['home_off'].replace(['1.87'], '2.06')
        final_2['home_def'] = final_2['home_def'].replace(['0.9'], '0.81')
        final_2['home_spi'] = final_2['home_spi'].replace(['66.62'], '72')
        final_2['home_rank'] = final_2['home_rank'].replace(['41'], '28')
        final_2['away_team'] = final_2['away_team'].replace(['Poland'], ' ')
        final_2['away_off'] = final_2['away_off'].replace(['2.06'], '0')
        final_2['away_def'] = final_2['away_def'].replace(['0.81'], '0')
        final_2['away_spi'] = final_2['away_spi'].replace(['72'], '0')
        final_2['away_rank'] = final_2['away_rank'].replace(['28'], '0')
        
    #add correct away team to finals
    if (scores[1][1] > scores[1][1][1]).any():
        final_2['away_team'] = final_2['away_team'].replace([' '], 'Sweden')
        final_2['away_off'] = final_2['away_off'].replace(['0'], '2.23')
        final_2['away_def'] = final_2['away_def'].replace(['0'], '0.78')
        final_2['away_spi'] = final_2['away_spi'].replace(['0'], '75.38')
        final_2['away_rank'] = final_2['away_rank'].replace(['0'], '24')
    else:
        final_2['away_team'] = final_2['away_team'].replace([' '], 'Czech Republic')
        final_2['away_off'] = final_2['away_off'].replace(['0'], '2.27')
        final_2['away_def'] = final_2['away_def'].replace(['0'], '0.68')
        final_2['away_spi'] = final_2['away_spi'].replace(['0'], '78.03')
        final_2['away_rank'] = final_2['away_rank'].replace(['0'], '19')
        
    #remove data not required 
    final_2_sin = final_2.select_dtypes(include=['int', 'float'])
    final_2_sin = final_2_sin.drop(columns=['match', 'home_score', 'away_score'])
    
    #predict score
    finals_score = []
    finals_score.append(predict_match(home_clf, final_2_sin))
    finals_score.append(predict_match(away_clf, final_2_sin))
    
    #show match result
    home_team = final_2['home_team']
    away_team = final_2['away_team']
    print('')
    print('{} {} : {} {}'.format(home_team[3], finals_score[0][0], finals_score[1][0], away_team[3]))
    print('')
    
    #show qualified team
    if (finals_score > finals_score[1]).any():
        print('{} has qualified!'.format(home_team[3]))
        winners.append(home_team[3])
    else:
        print('{} has qualified!'.format(away_team[3]))
        winners.append(away_team[3])
    print('')
        
    #finals 3
    final_3 = pd.DataFrame()
    final_3 = uefa_matches.drop([0,1,2,3,4,5,7,8])
    
    #show outcome of first two matches on branch
    home_team = uefa_matches['home_team']
    away_team = uefa_matches['away_team']
    print('')
    print('{} {} : {} {}'.format(home_team[6], scores[2][0][0], scores[2][0][1], away_team[6]))
    print('')
    print('{} {} : {} {}'.format(home_team[7], scores[2][1][0], scores[2][1][1], away_team[7]))
    print('')
    
    #add correct home team to finals
    if (scores[2][0] > scores[2][0][1]).any():
        final_3['away_team'] = final_3['away_team'].replace(['Turkey'], ' ')
        final_3['away_off'] = final_3['away_off'].replace(['1.95'], '0')
        final_3['away_def'] = final_3['away_def'].replace(['1.03'], '0')
        final_3['away_spi'] = final_3['away_spi'].replace(['65'], '0')
        final_3['away_rank'] = final_3['away_rank'].replace(['44'], '0')
    else:
        final_3['home_team'] = final_3['home_team'].replace(['Portugal'], 'Turkey')
        final_3['home_off'] = final_3['home_off'].replace(['2.85'], '1.95')
        final_3['home_def'] = final_3['home_def'].replace(['0.54'], '1.03')
        final_3['home_spi'] = final_3['home_spi'].replace(['87.35'], '65')
        final_3['home_rank'] = final_3['home_rank'].replace(['6'], '44')
        final_3['away_team'] = final_3['away_team'].replace(['Turkey'], ' ')
        final_3['away_off'] = final_3['away_off'].replace(['1.95'], '0')
        final_3['away_def'] = final_3['away_def'].replace(['1.03'], '0')
        final_3['away_spi'] = final_3['away_spi'].replace(['65'], '0')
        final_3['away_rank'] = final_3['away_rank'].replace(['44'], '0')
        
    #add correct away team to finals
    if (scores[2][1] > scores[2][1][1]).any():
        final_3['away_team'] = final_3['away_team'].replace([' '], 'Italy')
        final_3['away_off'] = final_3['away_off'].replace(['0'], '2.44')
        final_3['away_def'] = final_3['away_def'].replace(['0'], '0.48')
        final_3['away_spi'] = final_3['away_spi'].replace(['0'], '84.36')
        final_3['away_rank'] = final_3['away_rank'].replace(['0'], '10')
    else:
        final_3['away_team'] = final_3['away_team'].replace([' '], 'North Macedonia')
        final_3['away_off'] = final_3['away_off'].replace(['0'], '1.69')
        final_3['away_def'] = final_3['away_def'].replace(['0'], '0.98')
        final_3['away_spi'] = final_3['away_spi'].replace(['0'], '61.22')
        final_3['away_rank'] = final_3['away_rank'].replace(['0'], '52')
        
    #remove data not required
    final_3_sin = final_3.select_dtypes(include=['int', 'float'])
    final_3_sin = final_3_sin.drop(columns=['match', 'home_score', 'away_score'])
    
    #predict score
    finals_score = []
    finals_score.append(predict_match(home_clf, final_3_sin))
    finals_score.append(predict_match(away_clf, final_3_sin))
    
    #show result
    home_team = final_3['home_team']
    away_team = final_3['away_team']
    print('')
    print('{} {} : {} {}'.format(home_team[6], finals_score[0][0], finals_score[1][0], away_team[6]))
    print('')
    
    #show qualified team
    if (finals_score > finals_score[1]).any():
        print('{} has qualified!'.format(home_team[6]))
        winners.append(home_team[6])
    else:
        print('{} has qualified!'.format(away_team[6]))
        winners.append(away_team[6])
    print('')
    
    #return the three teams that qualified
    return winners

#create lists to store winners in, one for each classifier type used and one combined
#this will be used for graphs
qualified_teams = []    #combined
lg_qt = []              #logistic regression
dtc_qt = []             #decision tree
svc_qt = []             #support vector machine
km_qt = []              #k-means

#iterate for how times you want to predict, used for graphing results
for i in range(iterations):
    #-----------------------------------------------------------------------------------------------------------------------
    #   home score
    #-----------------------------------------------------------------------------------------------------------------------
    
    #Split into train and test data (home score)
    h_X_train, h_X_test, h_y_train, h_y_test = train_test_split(X_all, y_hs_all, 
                                                                test_size = 100,
                                                                random_state = 15)
    
    #logistic regression
    y_pred_lg = train_predict(h_lg, h_X_train, h_y_train, h_X_test, h_y_test)
    # print('')
    # print('LG')
    # print(pd.DataFrame(data=np.c_[h_y_test, y_pred_lg]))
    # print('')
    
    # #SVC
    y_pred_svc = train_predict(h_svc, h_X_train, h_y_train, h_X_test, h_y_test)
    # print('')
    # print('SVC')
    # print(pd.DataFrame(data=np.c_[h_y_test, y_pred_svc]))
    # print ('')
    
    #dtc
    y_pred_dtc = train_predict(h_dtc, h_X_train, h_y_train, h_X_test, h_y_test)
    # print('')
    # print('DTC')
    # print(pd.DataFrame(data=np.c_[h_y_test, y_pred_dtc]))
    # print ('')
    
    #k-means
    y_pred_km = train_predict(h_kmeans, h_X_train, h_y_train, h_X_test, h_y_test)
    # print('')
    # print('KMEANS')
    # print(pd.DataFrame(data=np.c_[h_y_test, y_pred_km]))
    # print('')
    
    #-----------------------------------------------------------------------------------------------------------------------
    #   away score
    #-----------------------------------------------------------------------------------------------------------------------
    
    #Split into train and test data (away score)
    a_X_train, a_X_test, a_y_train, a_y_test = train_test_split(X_all, y_as_all, 
                                                                test_size = 100,
                                                                random_state = 15)
    
    #logistic regression
    y_pred_lg = train_predict(a_lg, a_X_train, a_y_train, a_X_test, a_y_test)
    # print('')
    # print('LG')
    # print(pd.DataFrame(data=np.c_[a_y_test, y_pred_lg]))
    # print('')
    
    #SVC
    y_pred_svc = train_predict(a_svc, a_X_train, a_y_train, a_X_test, a_y_test)
    # print('')
    # print('SVC')
    # print(pd.DataFrame(data=np.c_[a_y_test, y_pred_svc]))
    # print ('')
    
    #dtc
    y_pred_dtc = train_predict(a_dtc, a_X_train, a_y_train, a_X_test, a_y_test)
    # print('')
    # print('DTC')
    # print(pd.DataFrame(data=np.c_[a_y_test, y_pred_dtc]))
    # print ('')
    
    #k-means
    y_pred_km = train_predict(a_kmeans, a_X_train, a_y_train, a_X_test, a_y_test)
    # print('')
    # print('KMEANS')
    # print(pd.DataFrame(data=np.c_[a_y_test, y_pred_km]))
    # print('')
    
    #predict remaining uefa qualifiers using: Decision Tree Classifier
    dtc_results = predict_uefa(h_dtc, a_dtc, "DTC")
    dtc_qt.append(dtc_results)
    qualified_teams.append(dtc_results)
    
    #predict remaining uefa qualifiers using: Logistic Regression
    lg_results = predict_uefa(h_lg, a_lg, "LG")
    lg_qt.append(lg_results)
    qualified_teams.append(lg_results)
    
    #predict remaining uefa qualifiers using: K-Means
    km_results = predict_uefa(h_kmeans, a_kmeans, "KMEANS")
    km_qt.append(km_results)
    qualified_teams.append(km_results)
    
    #predict remaining uefa qualifiers using: Support Vector Machine
    svc_results = predict_uefa(h_svc, a_svc, "SVC")
    svc_qt.append(svc_results)
    qualified_teams.append(svc_results)
    
#counts the number of times each nation is predicted to qualify    
def count_winners(winners):
    #initialise value of all nations to 0
    austria = 0 
    czechia = 0
    italy = 0
    north_mcd = 0
    poland = 0
    portugal = 0
    russia = 0
    scotland = 0
    sweden = 0
    turkey = 0
    ukraine = 0
    wales = 0
    
    #increases appropriate values
    for set in winners:
        austria += set.count('Austria')
        czechia += set.count('Czech Republic')
        italy += set.count('Italy')
        north_mcd += set.count('North Macedonia')
        poland += set.count('Poland')
        portugal += set.count('Portugal')
        russia += set.count('Russia')
        scotland += set.count('Scotland')
        sweden += set.count('Sweden')
        turkey += set.count('Turkey')
        ukraine += set.count('Ukraine')
        wales += set.count('Wales')
        
    #returns a list of the quantifying values in alphabetical order
    return [austria, czechia, italy, north_mcd, poland, portugal, russia, scotland, sweden, turkey, ukraine, wales]
    

#positions of bars
x_coords = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

#height values of bars for each graph
#summary
heights_total = count_winners(qualified_teams)
#logistic regression
heights_lg = count_winners(lg_qt)
#decision tree classifier
heights_dtc = count_winners(dtc_qt)
#support vector machine
heights_svc = count_winners(svc_qt)
#k-means
heights_km = count_winners(km_qt)


labels = ['AUT', 'CZR', 'ITA', 'NMK', 'POL', 'POR', 'RUS', 
          'SCO', 'SWE', 'TUR', 'UKR', 'WAL']

#summary
plt.bar(x_coords, heights_total, tick_label=labels, width=5, color=['blue'])
plt.xlabel('Countries in UEFA World Cup 2022 Qualification Round 2')
plt.ylabel('Number of predicted qualifications')
plt.title('UEFA World Cup 2022 Qualification Round 2 Combined Prediction Results (Summary)')
plt.show()

#logistic regression
plt.bar(x_coords, heights_lg, tick_label=labels, width=5, color=['blue'])
plt.xlabel('Countries in UEFA World Cup 2022 Qualification Round 2')
plt.ylabel('Number of predicted qualifications')
plt.title('UEFA World Cup 2022 Qualification Round 2 Combined Prediction Results (Logistic Regression)')
plt.show()

#decision tree classifier
plt.bar(x_coords, heights_dtc, tick_label=labels, width=5, color=['blue'])
plt.xlabel('Countries in UEFA World Cup 2022 Qualification Round 2')
plt.ylabel('Number of predicted qualifications')
plt.title('UEFA World Cup 2022 Qualification Round 2 Combined Prediction Results (Decision Tree Classifier)')
plt.show()

#support vector machine
plt.bar(x_coords, heights_svc, tick_label=labels, width=5, color=['blue'])
plt.xlabel('Countries in UEFA World Cup 2022 Qualification Round 2')
plt.ylabel('Number of predicted qualifications')
plt.title('UEFA World Cup 2022 Qualification Round 2 Combined Prediction Results (Support Vector Machine)')
plt.show()

#k-means
plt.bar(x_coords, heights_km, tick_label=labels, width=5, color=['blue'])
plt.xlabel('Countries in UEFA World Cup 2022 Qualification Round 2')
plt.ylabel('Number of predicted qualifications')
plt.title('UEFA World Cup 2022 Qualification Round 2 Combined Prediction Results (K-Means)')
plt.show()