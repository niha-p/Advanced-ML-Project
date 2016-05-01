import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

pd.options.display.mpl_style = 'default'

#Get data from crime locations
crime = pd.read_csv("train.csv")
features = ['X','Y']


#Relationship between crimes DRUG/NARCOTIC and PROSTITUTION
sub_crime1 = crime[(crime['Category'] == "DRUG/NARCOTIC")]
sub_crime2 = crime[(crime['Category'] == "PROSTITUTION")]
crime_X1 = sub_crime1[features]
crime_X2 = sub_crime2[features]
frames = [crime_X1, crime_X2]
combo=pd.concat(frames)
print len(crime_X1)
print len(crime_X2)
print len(combo)


#DRUG/NARCOTIC crime cluster 
num_clusters=15
km = KMeans(num_clusters,init='k-means++')
crime_km_fit1 = km.fit(crime_X1)
ax = crime_X1.plot(kind='scatter',x='X',y='Y', legend=str(num_clusters), figsize=(8, 6))
pd.DataFrame(crime_km_fit1.cluster_centers_).plot(kind='scatter',x=0,y=1,color='k',ax=ax)
ax.set_title(str(num_clusters) + " Drug/Narcotic Clusters")
plt.show()
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.savefig('crime1.png')


#PROSTITUTION crime cluster 
num_clusters=15
km = KMeans(num_clusters,init='k-means++')
crime_km_fit2 = km.fit(crime_X2)
ax = crime_X2.plot(kind='scatter',x='X',y='Y', legend=str(num_clusters), figsize=(8, 6))
pd.DataFrame(crime_km_fit2.cluster_centers_).plot(kind='scatter',x=0,y=1,color='k',ax=ax)
ax.set_title(str(num_clusters) + " Prostitution Clusters")
plt.show()
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.savefig('crime2.png')


#Plotting both DRUG/NARCOTIC and PROSTITUION clusters together
crime_base = combo.plot(kind='scatter',x='X',y='Y', legend=str(num_clusters), figsize=(10, 8))
ax=pd.DataFrame(crime_km_fit1.cluster_centers_).plot(kind='scatter',x=0,y=1,color='r', ax=crime_base)
pd.DataFrame(crime_km_fit2.cluster_centers_).plot(kind='scatter',x=0,y=1,color='k',ax=ax)
crime_base.set_title("Clustering of Drug/Narcotic and Prostitution")
plt.show()
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.savefig('crime12.png')





