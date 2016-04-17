import csv
from sets import Set
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
categories={}
count=0
for row in data:
    if row[1]=='Category':
        continue
    if row[2] in categories:
        continue
    else:
        categories[row[2]]=count
        count=count+1
##for c in sorted(categories):
##    print c
print len(categories)
