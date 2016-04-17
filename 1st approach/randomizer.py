import csv
from sets import Set
import json
import random
import codecs

def load_data(file_path):
  """
  This method reads the dataset, and returns a list of rows.
  Each row is a list containing the values in each column.
  """
  import csv
  with file(file_path) as f:
    dialect = csv.Sniffer().sniff(f.read(2048))
    f.seek(0)
    reader = csv.reader(f, dialect)
    return [l for l in reader]


dataX = load_data('train-partial.csv')
dataY=[]
f=codecs.open('Yt.csv',"rb","utf-16")
csvread=csv.reader(f,delimiter='\t')
csvread.next()
for row in csvread:
    dataY.append(int(row[0]))

arr = random.sample(range(1, 878017), 100000)

r=len(arr)


writeX=open('train_small.csv', "wb")
writeY=open('Y_small.csv', "wb")

writerX = csv.writer(writeX)
writerY = csv.writer(writeY)
print arr[0],arr[1],arr[2],arr[3],arr[4]
for x in xrange(r):
    
    writerX.writerow(dataX[arr[x]])
    writerY.writerow([dataY[arr[x]]])

writeX.close()
writeY.close()
