import os
import time
from collections import defaultdict 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score, accuracy_score, make_scorer, f1_score
from scipy.stats import ttest_ind
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


#Reading test and training set
train_file = './dataset/UNSW_NB15_training-set.csv'
test_file = './dataset/UNSW_NB15_testing-set.csv'

train_df = pd.read_csv(train_file, index_col = 0) # Reading training set from file
train_header_names = train_df.keys() #Get the names of columns
train_df.describe()

test_df = pd.read_csv(test_file, index_col = 0) # Reading test set from file
test_header_names = test_df.keys() #Get the names of columns
test_df.describe()

# TRAINING SET: Differentiating between nominal, binary, and numeric features

train_col_names = np.array(train_header_names)

train_nominal_idx = [1, 2, 3, 42]
train_binary_idx = [41, 43] 
train_numeric_idx = list(set(range(43)).difference(train_nominal_idx).difference(train_binary_idx))

train_nominal_cols = train_col_names[train_nominal_idx].tolist()
train_binary_cols = train_col_names[train_binary_idx].tolist()
train_numeric_cols = train_col_names[train_numeric_idx].tolist()

# TEST SET: Differentiating between nominal, binary, and numeric features

test_col_names = np.array(test_header_names)

test_nominal_idx = [1, 2, 3, 42]
test_binary_idx = [41, 43] 
test_numeric_idx = list(set(range(43)).difference(test_nominal_idx).difference(test_binary_idx))

test_nominal_cols = test_col_names[test_nominal_idx].tolist()
test_binary_cols = test_col_names[test_binary_idx].tolist()
test_numeric_cols = test_col_names[test_numeric_idx].tolist()

#Data distribution of training set 

train_label = train_df['label'].value_counts()
train_attack_cats = train_df['attack_cat'].value_counts()

print(train_label)
print('\n')
print(train_attack_cats)

#Plotting attack categories istances of training set
train_attack_cats.plot(kind='bar', figsize=(10,5), fontsize=20)

#Plotting attack labels (0: normal, 1: attack) of training set
train_label.plot(kind='bar', figsize=(10,5), fontsize=15)

#Data distribution of test set 

test_label = test_df['label'].value_counts()
test_attack_cats = test_df['attack_cat'].value_counts()

print(test_label)
print('\n')
print(test_attack_cats)

#Plotting attack categories istances of test set
test_attack_cats.plot(kind='bar', figsize=(10,5), fontsize=20)

#Plotting attack labels (0: normal, 1: attack) of test set
test_label.plot(kind='bar', figsize=(10,5), fontsize=15)

# Binary features: by definition, all of these features should have a min of 0.0 and a max of 1.0

train_df[train_binary_cols].describe().transpose()

#Check if some columns has always the same value (to see if some features can be removed)

# Time complexity: O(n)
def unique_cols(train_df):
    a = train_df.to_numpy()
    return (a[0] == a).all(0)

unique_cols(train_df)

#Removing class label and attack_cat feature from training set 

train_y = train_df['label']
train_x_raw = train_df.drop(['attack_cat','label'], axis=1)
train_nominal_cols.remove('attack_cat')

#Remove nan records
train_x_raw.dropna(axis=0, inplace=True)

# Transform nominal features in binary features
train_x = pd.get_dummies(train_x_raw, columns=train_nominal_cols, drop_first=True) 
dummy_variables = list(set(train_x)-set(train_x_raw))

train_x.describe()


#Removing class label and attack_cat feature from test set 

test_y = test_df['label']
test_x_raw = test_df.drop(['attack_cat','label'], axis=1)
test_nominal_cols.remove('attack_cat')

test_x = pd.get_dummies(test_x_raw, columns=test_nominal_cols, drop_first=True)
dummy_variables = list(set(test_x)-set(test_x_raw))
test_x.describe()

#Adding missing binary features at 0 in test set
for column in train_x.columns:
    if column not in test_x.columns:
        test_x[column] = 0

#Removing features which are in test set but not in training set
for column in test_x.columns:
    if column not in train_x.columns:
        test_x.drop([column], axis=1, inplace=True)


test_x.describe()

#Check if test set and training set have the same number of features
print(len(train_x.keys()))

#Normalization of numeric features

#FIT
standard_scaler = StandardScaler().fit(train_x[train_numeric_cols])

#TRANSFORM
train_x[train_numeric_cols] = standard_scaler.transform(train_x[train_numeric_cols])
test_x[test_numeric_cols] = standard_scaler.transform(test_df[test_numeric_cols])

#Sorting test set and training set by features in alphabetical order
train_x = train_x.sort_index(axis=1)
test_x = test_x.sort_index(axis=1)

test_x.describe()

train_x.describe()

#Check if training set and test set have features in the same order
train_x.keys() == test_x.keys()



### Logistic Regression
model_lr = LogisticRegression(max_iter=1000)

start = time.time()
model_lr.fit(train_x, train_y)
end = time.time()
print('Training time: ' + str(round(end-start, 2)) + ' sec')

y_pred = model_lr.predict(test_x)
y_pred_proba_lr = model_lr.predict_proba(test_x)[:, 1]  # 获取正类的预测概率

# 计算AUC
print('Logistic Regression AUC: ', roc_auc_score(test_y, y_pred_proba_lr))


print(accuracy_score(test_y, y_pred))

#Classification report
print(classification_report(test_y,y_pred,target_names = ['normal','attack']))

#confusion matrix
ConfusionMatrixDisplay.from_predictions(test_y, y_pred,display_labels = ['normal','attack'])
plt.show()




# ### Neural Network
# model_nn = MLPClassifier(solver='adam', max_iter=1000, hidden_layer_sizes = (10,5))

# start = time.time()
# model_nn.fit(train_x, train_y)
# end = time.time()
# print('Training time: ' + str(round(end-start, 2)) + ' sec')

# y_pred = model_nn.predict(test_x)
# y_pred_proba_lr_nn = model_lr.predict_proba(test_x)[:, 1]  # 获取正类的预测概率

# # 计算AUC
# print('Neural Network AUC: ', roc_auc_score(test_y, y_pred_proba_lr_nn))

# print(accuracy_score(test_y, y_pred))

# #Classification report
# print(classification_report(test_y,y_pred,target_names = ['normal','attack']))

# #confusion matrix
# ConfusionMatrixDisplay.from_predictions(test_y, y_pred,display_labels = ['normal','attack'])
# plt.show()


