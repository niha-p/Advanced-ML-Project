import requests
import urllib, urllib2, json
import csv

fname="firestationLoc.txt"
#with open(fname) as f:
#    lines = f.readlines()
lines = [line.rstrip('\n') for line in open(fname)]

#print lines 
lat=[]
longi=[]

outputFile = open('fireStatLatLong.csv', 'w')
outputWriter = csv.writer(outputFile)


for i in lines:        
        url = 'http://maps.google.com/maps/api/geocode/json?' + urllib.urlencode({'address':i})
        response = urllib2.urlopen(url)
        result = json.load(response)
        try:
                l2=result['results'][0]['geometry']['location']['lat']
                l1=result['results'][0]['geometry']['location']['lng']
                outputWriter.writerow([l1,l2])

        except:
                print ("Not found: "+i)	
                outputWriter.writerow(['To find: '+i,'To find: '+i])




outputFile.close()                