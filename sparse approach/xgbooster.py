import pandas as pd
import csv
import xgboost as xgb
import logging
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
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


params = {'max_depth':8, 'eta':0.05, 'silent':1,
              'objective':'multi:softprob', 'num_class':39, 'eval_metric':'mlogloss',
              'min_child_weight':3, 'subsample':0.6,'colsample_bytree':0.6, 'nthread':4}
num_rounds = 250
t1=time.time()
xgbtrain = xgb.DMatrix(training[features], label=training['crime'])
classifier = xgb.train(params, xgbtrain, num_rounds)
t2=time.time()
score = log_loss(validation['crime'].values(), classifier.predict(validation[features]))
print 'Gradient Booster Tree Loss and Time:'
print score,(t2-t1)
