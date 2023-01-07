import csv
import numpy as np
#import xlsxwriter
import pandas as pd

input_file = "byteTrackPbcourt_metadata.csv"

## Read in per-frame csv file
with open(input_file) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	InPlay = []
	InServicePos = []
	list2 = []
	for row in readCSV:
		list2.append(row)
	for i in range(1, len(list2)):
		InPlay += [float(list2[i][1])]
		InServicePos += [float(list2[i][2])]

InBoth = [float(bool(InPlay[i]) and bool(InServicePos[i])) for i in range(len(InServicePos))]  
## Downsample to per-second, assuming 30fps
W = 30
thelen = len(InPlay)
InPlaySec = [np.mean(InPlay[i:i+W]) for i in range(0, thelen, W)]
InServSec = [np.mean(InServicePos[i:i+W]) for i in range(0, thelen, W)]
InBothSec = [np.mean(InBoth[i:i+W]) for i in range(0, thelen, W)]

times = list(range(0, len(InServSec)))

#print(len(times))
#print(len(InPlaySec))
#print(len(InServSec))
#print(times[0:30])
print(InPlaySec[0:30])
print(InServSec[0:30])
print(InBothSec[0:30])
## combine into one dataframe

zipped = list(zip(times, InBothSec, InPlaySec, InServSec))
df = pd.DataFrame(zipped, columns=['time[s]', 'InBoth', 'InPlay', 'Serving'])
#print(df[1:20])
## Write out xls file

df.to_excel('pboutput.xlsx')#, index=False)
