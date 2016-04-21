import pandas as pd
import numpy as np
import logging
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA

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
'TARAVAL', 'TENDERLOIN','X','Y']
 
features2 = [x for x in range(0,24)]
features = features + features2

"""
print "Variance Threshold"
sel = VarianceThreshold(threshold=(0.90 * (1 - 0.90)))
selector=sel.fit(training[features])
print selector.get_support(indices=True)

for i in range(0,len(features)):
    if i in selector.get_support(indices=True):
        print features[i]


print "Select from Model - Logistic"
modelLReg = LogisticRegression()
modelLReg = modelLReg.fit(training[features], training['crime'])
model = SelectFromModel(modelLReg, prefit=True)
print model.get_support(indices=True)

for i in range(0,len(features)):
    if i in model.get_support(indices=True):
        print features[i]


print "Tree Based Feature Selection"
clf = ExtraTreesClassifier()
clf = clf.fit(training[features], training['crime'])
model = SelectFromModel(clf, prefit=True)
print model.get_support(indices=True)

for i in range(0,len(features)):
    if i in model.get_support(indices=True):
        print features[i]


print "Lasso CV"
clf = LassoCV()
clf=clf.fit(training[features], training['crime'])
model = SelectFromModel(clf, threshold=0.25,prefit=True)
print model.get_support(indices=True)

for i in range(0,len(features)):
    if i in model.get_support(indices=True):
        print features[i]


print "fclassif - Select Percentile"
selector = SelectPercentile(f_classif, percentile=50)
selector = selector.fit(training[features], training['crime'])
print selector.get_support(indices=True)

for i in range(0,len(features)):
    if i in selector.get_support(indices=True):
        print features[i]

"""
print "PCA"

"""
pca = PCA(n_components='mle')
selector=pca.fit(training[features])
print selector.get_params()

for i in range(0,len(features)):
    if i in selector.get_support(indices=True):
        print features[i]
"""

from matplotlib.mlab import PCA
res = PCA(training[features])
print "weights of input vectors: %s" % res.Wt


"""
*******  OUTPUT  *******
Variance Threshold
[ 0  1  2  3  4  5  6  7 10 11 14 18]
Select from Model - Logistic
[ 7  8  9 10 12 13 14 16 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 41 42]
Tree Based Feature Selection
[17 18]
Lasso CV
[ 7  8  9 11 13 14 15 16 17 19 20 21 24 25 26 30 31 32 33 34 37 39 40 41 42]
fclassif - Select Percentile
[ 7  8  9 10 11 12 13 14 15 16 17 19 20 21 22 25 26 27 31 39 41]

"""



"""
*****Doesn't work - Only for non-negative features ****
print "chi2 - Select Percentile"
selector = SelectPercentile(chi2, percentile=75)
selector = selector.fit(training[features], training['crime'])
print selector.get_support(indices=True)
"""

"""
*****Doesn't work*****
print "Select KBest"
X_new = SelectKBest(chi2, k=20)
selector=X_new.fit_transform(training[features])
#print selector.get_support(indices=True)
print selector.shape
"""


"""
--------OUTPUT--------

Variance Threshold
[ 0  1  2  3  4  5  6  7 10 11 14 18]
Monday
Tuesday
Wednesday
Thursday
Friday
Saturday
Sunday
BAYVIEW
MISSION
NORTHERN
SOUTHERN
Y
Select from Model - Logistic
[ 7  8  9 10 12 13 16 18 19 20 21 22 23 24 25 26 27 28 29 30 31 41 42]
BAYVIEW
CENTRAL
INGLESIDE
MISSION
PARK
RICHMOND
TENDERLOIN
Y
0
1
2
3
4
5
6
7
8
9
10
11
12
22
23
Tree Based Feature Selection
[17 18]
X
Y
Lasso CV
[ 7  8  9 11 13 14 15 16 17 20 23 24 25 26 30 31 32 33 34 37 39 40 41 42]
BAYVIEW
CENTRAL
INGLESIDE
NORTHERN
RICHMOND
SOUTHERN
TARAVAL
TENDERLOIN
X
1
4
5
6
7
11
12
13
14
15
18
20
21
22
23
fclassif - Select Percentile
[ 7  8  9 10 11 12 13 14 15 16 17 19 20 21 22 24 25 26 27 31 41]
BAYVIEW
CENTRAL
INGLESIDE
MISSION
NORTHERN
PARK
RICHMOND
SOUTHERN
TARAVAL
TENDERLOIN
X
0
1
2
3
5
6
7
8
12
22



"""



