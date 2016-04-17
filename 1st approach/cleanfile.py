import csv
from sets import Set
import json
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


data = load_data('train.csv')
do=open('dictionaries.txt','wb')
categories={}
districts={}
resolutions={}
descriptions={}
days={'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
countr=0
countd=0
count=0
countdes=0
for row in data:
    if row[1]=='Category':
        continue
    if row[1] not in categories:
        categories[row[1]]=count
        count=count+1

    if row[4] not in districts:
        districts[row[4]]=countd
        countd=countd+1
        
    if row[5] not in resolutions:
        resolutions[row[5]]=countr
        countr=countr+1
    if row[2] not in descriptions:
        descriptions[row[2]]=countdes
        countdes=countdes+1
        
fo=csv.writer(open('train_2.csv','wb'))
for bits in data:
    if bits[1]=='Category':
        bits.append('Day')
        bits.append('Month')
        bits.append('Year')
        bits.append('Hour')
        bits.append('Minutes')
        bits.append('Seconds')
        fo.writerow(bits)
    else:
        datetime=bits[0].split(' ')
        date=datetime[0].split('-')
        time=datetime[1].split(':')
        dd=date[2]
        mm=date[1]
        yy=date[0]
        hh=time[0]
        mn=time[1]
        sec=time[2]
        bits[1]=str(categories[bits[1]])
        bits[2]=str(descriptions[bits[2]])
        bits[3]=str(days[bits[3]])
        bits[4]=str(districts[bits[4]])
        bits[5]=str(resolutions[bits[5]])
        bits.append(dd)
        bits.append(mm)
        bits.append(yy)
        bits.append(hh)
        bits.append(mn)
        bits.append(sec)
        fo.writerow(bits)
do.write('Categories\n\n')
do.write(json.dumps(categories, indent=4, sort_keys=True))
do.write('\n\n\nDistricts\n\n')
do.write(json.dumps(districts, indent=4, sort_keys=True))
do.write('\n\n\nResolutions\n\n')
do.write(json.dumps(resolutions, indent=4, sort_keys=True))
do.write('\n\n\nDescriptions\n\n')
do.write(json.dumps(descriptions, indent=4, sort_keys=True))
do.close()
