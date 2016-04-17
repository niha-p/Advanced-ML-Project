import csv
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import codecs
from sklearn.externals import joblib

csvfile = open('train_small.csv', 'rb')
#csvfiley = open('Yt.csv', 'rb')
X=[]
Y=[]

fieldnames = ("DayOfWeek","PdDistrict","X","Y","Day","Month","Year","Hour","Minutes")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    for r in row:
        row[r]=float(row[r])
    X.append(row)


f=codecs.open('Y_small.csv',"rb")
csvread=csv.reader(f,delimiter='\t')
for row in csvread:
    Y.append(int(row[0]))
    
print Y[0],Y[1],Y[2],Y[3],Y[4]        
print len(X),len(Y)

def features(d):
    return d
vect = DictVectorizer()
X_train = vect.fit_transform(features(d)for d in X)

model = LogisticRegression(tol=0.01)
model.fit(X_train, Y)
joblib.dump(model, 'my_model.pkl', compress=9)
##model= joblib.load('my_model.pkl')
print 'model fitted'
tests=[]
c=0
for tr in X:
    trans=vect.transform(tr)
    print model.predict(trans)
    c=c+1
    if c==100:
        break

##t={"DayOfWeek":2.0,"PdDistrict":0.0,"X":-122.4258916751,"Y":37.7745985957,"Day":13.0,"Month":5.0,"Year":2015.0,"Hour":23.0,"Minutes":53.0}
##test=vect.transform(t)
##print model.predict(test)
