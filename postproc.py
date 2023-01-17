import csv
import numpy as np
#import xlsxwriter
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

input_file = "byteTrackPbcourt_metadata.csv"
vid = "byteTrackPbcourt_trajectory.mp4"

video = cv2.VideoCapture(vid);
fps = video.get(cv2.CAP_PROP_FPS)
print("fps",fps)
fps = int(fps)
print("fpsint",fps)
video.release()

def convert(seconds, l = False):
	seconds = seconds % (24 * 3600)
	hour = seconds // 3600
	seconds %= 3600
	minutes = seconds // 60
	seconds %= 60
	if l:
		return "%d:%02d:%02d.001" % (hour,minutes, seconds)
	else:
		return "%02d:%02d" % (minutes, seconds)

## Read in per-frame csv file
with open(input_file) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	InPlay = []
	LTDR = []
	InServicePos = []
	list2 = []
	for row in readCSV:
		list2.append(row)
	for i in range(1, len(list2)):
		LTDR += [float(list2[i][1])]
		InPlay += [float(list2[i][2])]
		InServicePos += [float(list2[i][3])]

InBoth = [float(bool(InPlay[i]) and bool(InServicePos[i])) for i in range(len(InServicePos))]  
## Downsample to per-second, assuming 30fps
#W = 30
#W = 60
W=fps
thelen = len(InPlay)
InPlaySec = [np.mean(InPlay[i:i+W]) for i in range(0, thelen, W)]
InServSec = [np.mean(InServicePos[i:i+W]) for i in range(0, thelen, W)]
LTDRSec = [np.mean(LTDR[i:i+W]) for i in range(0, thelen, W)]

state = 0
transcript = []
for i in range(0,len(InServSec)):
	#print(state,i)
	if state == 0 and InServSec[i] >= 1:
		state = 1
		#transcript += [["%s Serve" % (convert(i,True))]]
		transcript += [[convert(i,True)]]
		transcript += [["Serve"]]
		transcript += [[" "]]
	elif state == 1 and InServSec[i] <= 0.5:
		state = 0
	#print(state, i)

print(transcript)
with open('serveTranscript.csv', 'w') as csvfile: 
	# creating a csv writer object 
	csvwriter = csv.writer(csvfile) 
        
	# writing the fields 
	csvwriter.writerows(transcript) 

#InServSecSmooth = signal.savgol_filter(InServSec, 7, 3)
InServSecSmooth = np.convolve(InServSec, np.ones(7) / 7, mode='same')
x_=[]
for i in range(0,len(InServSec)):
	x_.append(convert(i))

plt.plot(x_, InServSec)
plt.plot(x_,InServSecSmooth, color='green')
plt.xlim([60, 120])
plt.xticks(rotation=45, ha="right")
#plt.show()

times = list(range(0, len(InServSec)))

#print(len(times))
#print(len(InPlaySec))
#print(len(InServSec))
#print(times[0:30])
print(InPlaySec[0:30])
print(InServSec[0:30])
print(LTDRSec[0:30])
## combine into one dataframe

zipped = list(zip(times, LTDRSec, InPlaySec, InServSec))
df = pd.DataFrame(zipped, columns=['time[s]', 'LTDR', 'InPlay', 'Serving'])
#print(df[1:20])
## Write out xls file

df.to_excel('pboutput.xlsx')#, index=False)
