import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt

#Enter the path to bank-additional-full.csv here
train = pd.read_csv("InputFolderPath/bank-additional-full.csv", delimiter=';')

label = LabelEncoder()
train['y'] = label.fit_transform(train['y'])

###Checking correlation amongst variables and with target variable
print(train.corr())

# Exploring categorical variables- looking at means here
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25,15))
train.groupby('job')['y'].mean().plot.bar(ax=axes[0,0])
train.groupby('marital')['y'].mean().plot(ax=axes[0,1])
train.groupby('education')['y'].mean().plot(ax=axes[1,0])
train.groupby('default')['y'].mean().plot(ax=axes[1,1])

# Exploring remaining categorical variables
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(25,15))
train.groupby('housing')['y'].mean().plot.bar(ax=axes[0,0])
train.groupby('loan')['y'].mean().plot(ax=axes[0,1])
train.groupby('contact')['y'].mean().plot(ax=axes[1,0])
train.groupby('month')['y'].mean().plot(ax=axes[1,1])
train.groupby('day_of_week')['y'].mean().plot(ax=axes[2,0])
train.groupby('poutcome')['y'].mean().plot(ax=axes[2,1])



# Exploring categorical variables - looking at the counts
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(25,15))
train.groupby('job')['y'].count().plot.bar(ax=axes[0,0])
train.groupby('marital')['y'].count().plot(ax=axes[0,1])
train.groupby('education')['y'].count().plot(ax=axes[1,0])
train.groupby('default')['y'].count().plot(ax=axes[1,1])
train.groupby('loan')['y'].count().plot(ax=axes[2,0])
train.groupby('contact')['y'].count().plot(ax=axes[2,1])
train.groupby('month')['y'].count().plot(ax=axes[3,0])
train.groupby('day_of_week')['y'].count().plot(ax=axes[3,1])
train.groupby('poutcome')['y'].count().plot(ax=axes[4,0])

# Preparing dataset - Use Tree Based classifier - Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Removing duration
train.drop('duration', axis=1, inplace=True)

X = train.ix[:, train.columns != 'y']
y = train['y']

# Converting category to binary features and scaling non-object variables
feat = []
for col in X.columns:
    if X[col].dtype != object:
        feat.append(col)
    if X[col].dtype == object:
        print("Encoding ", col)
        store = pd.get_dummies(X[col])
        X.drop(col, inplace=True, axis=1)
        X = pd.concat([X, store], axis=1)

minmax = MinMaxScaler()
X[feat] = minmax.fit_transform(X[feat])
print(X.head(5))

model1 = DecisionTreeClassifier()
score = cross_val_score(model1, X, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

model2 = LogisticRegression()
score = cross_val_score(model2, X, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

model4 = KNeighborsClassifier()
score = cross_val_score(model4, X, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

#Removing macroeconomic variables as they typically add noise. They have long horizon and to judge the impact large number
# of data points are needed
X.drop(["emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m", "nr.employed"], axis=1, inplace=True)

# Trying to work forward with Logistic Regression to improve F1 score using Recursive Feature Elimination
# Identifying top 20 features
from sklearn.feature_selection import RFE
rfe = RFE(LogisticRegression(),20)

rfe.fit(X, y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })
print(feature_rfe_scoring.sort_values(by='score', ascending=True))


feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values

X_rfe_20 = X[feat_rfe_20]
model = LogisticRegression()
score = cross_val_score(model, X_rfe_20, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

#decent boost testing reducing the number of variables to 10
from sklearn.feature_selection import RFE
rfe = RFE(LogisticRegression(),10)

rfe.fit(X, y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })
print(feature_rfe_scoring.sort_values(by='score', ascending=True))
feat_rfe_10 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values

X_rfe_10 = X[feat_rfe_10]
model = LogisticRegression()
score = cross_val_score(model, X_rfe_10, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

#decent boost testing reducing the number of variables to 5
from sklearn.feature_selection import RFE
rfe = RFE(LogisticRegression(),5)

rfe.fit(X, y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })
print(feature_rfe_scoring.sort_values(by='score', ascending=True))
feat_rfe_5 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values

X_rfe_5 = X[feat_rfe_5]
model = LogisticRegression()
score = cross_val_score(model, X_rfe_5, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

#decent boost testing reducing the number of variables to 5
from sklearn.feature_selection import RFE
rfe = RFE(LogisticRegression(),3)

rfe.fit(X, y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })
print(feature_rfe_scoring.sort_values(by='score', ascending=True))
feat_rfe_3 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values

X_rfe_3 = X[feat_rfe_3]
model = LogisticRegression()
score = cross_val_score(model, X_rfe_3, y, scoring='f1', cv=5)
print("Mean cross validation score F1 score", score.mean())
score = cross_val_score(model, X_rfe_3, y, scoring='accuracy', cv=5)
print("Mean cross validation score Accuracy", score.mean())

#using same set of variables for SVC
model = SVC() ## Extremely slow will try later
score = cross_val_score(model, X_rfe_3, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

# Implementing advanced tree based algorithm GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
score = cross_val_score(model, X_rfe_3, y, scoring='f1', cv=5)
print("Mean cross validation score", score.mean())

# Selecting variables basis our intuition
X = train.ix[:, train.columns != 'y']
y = train['y']

# Converting category to binary features and scaling non-object variables
feat =[]
for col in X.columns:
    if X[col].dtype != object:
        feat.append(col)
    if X[col].dtype == object:
        print("Encoding ", col)
        store = pd.get_dummies(X[col])
        X.drop(col, inplace=True, axis=1)
        X = pd.concat([X, store], axis=1)

features = ["retired", "student", "single", "university.degree", "no", "thu", "wed", "tue",
            "success", "failure", "cellular", "pdays", "previous"]

X_feat=X[features]
model = LogisticRegression()
score = cross_val_score(model, X_feat, y, scoring='f1', cv=5)
print("Mean cross validation score-F1 score", score.mean())
score = cross_val_score(model, X_feat, y, scoring='accuracy', cv=5)
print("Mean cross validation score-Accuracy", score.mean())




