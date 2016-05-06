import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.svm import SVC
import numpy as np
import logging
import time

logging.captureWarnings(True)

train_X_Y = pd.read_csv('train.csv', parse_dates = ['Dates'])

crime_model = preprocessing.LabelEncoder()
crime_data = crime_model.fit_transform(train_X_Y.Category)

days = pd.get_dummies(train_X_Y.DayOfWeek)
district = pd.get_dummies(train_X_Y.PdDistrict)
hour = train_X_Y.Dates.dt.hour
hour = pd.get_dummies(hour)

train_data = pd.concat([hour, days, district], axis=1)
train_data['crime']=crime_data
train_data['Y']=train_X_Y['Y']
train_data['X']=train_X_Y['X']

training, validation = train_test_split(train_data, train_size=.60)


features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 
'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION','NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN',
'TARAVAL', 'TENDERLOIN',
'X','Y']
 
features2 = [x for x in range(0,24)]
features = features + features2


##t1=time.time()
##modelBern = BernoulliNB()
##modelBern.fit(training[features], training['crime'])
##t2=time.time()
##predicted = np.array(modelBern.predict_proba(validation[features]))
##print 'Bernoulli Loss and Time:'
##print log_loss(validation['crime'], predicted),(t2-t1)
##
##t1=time.time()
##modelLReg = LogisticRegression(C=.01)
##modelLReg.fit(training[features], training['crime'])
##t2=time.time()
##predicted = np.array(modelLReg.predict_proba(validation[features]))
##print 'Logistic Regression Loss and Time:'
##print log_loss(validation['crime'], predicted),(t2-t1)
##
##t1=time.time()
##modelTree = DecisionTreeClassifier(max_depth=7)
##modelTree.fit(training[features], training['crime'])
##t2=time.time()
##predicted = np.array(modelTree.predict_proba(validation[features]))
##print 'Decision Tree Loss and Time:'
##print log_loss(validation['crime'], predicted),(t2-t1)

##t1=time.time()
##modelGauss = GaussianNB()
##modelGauss.fit(training[features], training['crime'])
##t2=time.time()
##predicted = np.array(modelGauss.predict_proba(validation[features]))
##print 'Gaussian Loss and Time:'
##print log_loss(validation['crime'], predicted),(t2-t1)

t1=time.time()
modelGr = GradientBoostingClassifier()
modelGr.fit(training[features], training['crime'])
t2=time.time()
predicted = np.array(modelGr.predict_proba(validation[features]))
print 'Gradient Boosted Loss and Time:'
print log_loss(validation['crime'], predicted),(t2-t1)


##modelKnn = KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=100, p=10, metric='minkowski')
##modelKnn.fit(training[features], training['crime'])
##predicted = np.array(modelKnn.predict_proba(validation[features]))
##print 'KNN Loss:'
##print log_loss(validation['crime'], predicted)

##t1=time.time()
##modelSVC = SVC(max_iter=1000)
##modelSVC.fit(training[features], training['crime'])
##t2=time.time()
##predicted = np.array(modelSVC.predict_proba(validation[features]))
##print 'SVC Loss and Time:'
##print log_loss(validation['crime'], predicted),(t2-t1)



