import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

pd.options.display.mpl_style = 'default'


#Get data from fire Station locations and crime locations
fireStation = pd.read_csv("fireStatLatLong.csv")
crime = pd.read_csv("train.csv")
features = ['X','Y']


#Fire Station Clusters
fireStation_X = fireStation[features]
num_clusters=15
km = KMeans(num_clusters,init='k-means++')
fireStat_km_fit = km.fit(fireStation_X)
ax = fireStation_X.plot(kind='scatter',x='X',y='Y', legend=str(num_clusters), figsize=(8, 6))
pd.DataFrame(fireStat_km_fit.cluster_centers_).plot(kind='scatter',x=0,y=1,color='k',ax=ax)
ax.set_title(str(num_clusters) + " Fire Station Clusters")
plt.show()
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.savefig('FireStationClusters.png')


#Crime Clusters
sub_crime = crime[(crime['Category'] == "ARSON")]
#print sub_crime
crime_X = sub_crime[features]
num_clusters=15
km = KMeans(num_clusters,init='k-means++')
crime_km_fit = km.fit(crime_X)
ax = crime_X.plot(kind='scatter',x='X',y='Y', legend=str(num_clusters), figsize=(8, 6))
pd.DataFrame(crime_km_fit.cluster_centers_).plot(kind='scatter',x=0,y=1,color='k',ax=ax)
ax.set_title(str(num_clusters) + " Arson Clusters")
plt.show()
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.savefig('CrimeClusters.png')


#Plotting both fire Station and Crime Clusters together
crime_base = crime_X.plot(kind='scatter',x='X',y='Y', legend=str(num_clusters), figsize=(10, 8))
ax=pd.DataFrame(crime_km_fit.cluster_centers_).plot(kind='scatter',x=0,y=1,color='r', ax=crime_base)
pd.DataFrame(fireStat_km_fit.cluster_centers_).plot(kind='scatter',x=0,y=1,color='k',ax=ax)
crime_base.set_title("Clustering of Crime(Arson) and Fire Station")
plt.show()
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.savefig('FireStat&Crime.png')





