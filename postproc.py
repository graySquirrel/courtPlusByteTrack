import csv
import numpy as np
#import xlsxwriter
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys, getopt
from PickleballScore import *

input_file = "byteTrackPbcourt_metadata.csv"
vid = "byteTrackPbcourt_trajectory.mp4"
starttimesec = 0
switchAtPoints = 0 # TODO

opts, args = getopt.getopt(sys.argv[1:],"s:",["startSec="])
for opt, arg in opts:
	if opt in ("-s", "--startSec"):
		starttimesec = int(arg)

print("starttimesec is",starttimesec)

video = cv2.VideoCapture(vid);
fps = video.get(cv2.CAP_PROP_FPS)
print("fps",fps)
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
	NearFar = []
	LeftRight = []
	list2 = []
	for row in readCSV:
		list2.append(row)
	for i in range(1, len(list2)):
		LTDR += [float(list2[i][1])]
		InPlay += [float(list2[i][2])]
		InServicePos += [float(list2[i][3])]
		NearFar += [float(list2[i][4])]
		LeftRight += [float(list2[i][5])]

InBoth = [float(bool(InPlay[i]) and bool(InServicePos[i])) for i in range(len(InServicePos))]  
## Downsample to per-second, assuming 30fps
#W = 30
#W = 60
print(fps)
W=int(round(fps,0))
print(W)
thelen = len(InPlay)
InPlaySec = [np.mean(InPlay[i:i+W]) for i in range(0, thelen, W)]
InServSec = [np.mean(InServicePos[i:i+W]) for i in range(0, thelen, W)]
LTDRSec = [np.mean(LTDR[i:i+W]) for i in range(0, thelen, W)]
NearFarSec = [np.mean(NearFar[i:i+W]) for i in range(0, thelen, W)]
LeftRightSec = [np.mean(LeftRight[i:i+W]) for i in range(0, thelen, W)]

state = 0
transcript = []

transcript += [[convert(starttimesec, True)]]
transcript += [["Start Of Game"]]
transcript += [[" "]]
pbs = PickleballScore()

for i in range(starttimesec,len(InServSec)):
	#print(state,i)
	# This is the event that creates a service annotation.  
	# Service annotation know far or near, and if server is on right or left as they see it.
	# If this is robust (and correct), we should be able to infer score...
	# Need to know if players switched at 6 or whatever point for accurate scoring.
	if state == 0 and InServSec[i] >= 1:
		state = 1
		NFtext = "Serve Far Side" if NearFarSec[i] < 1.5 else "Serve Near Side"
		LRtext = ", Left" if LeftRightSec[i] < 1.5 else ", Right"
		if NearFarSec[i] < 1.5 and LeftRightSec[i] < 1.5:
			pbs.update_state(ServerSide.FL)
		elif NearFarSec[i] < 1.5 and LeftRightSec[i] > 1.5:
			pbs.update_state(ServerSide.FR)
		elif NearFarSec[i] > 1.5 and LeftRightSec[i] < 1.5:
			pbs.update_state(ServerSide.NL)
		elif NearFarSec[i] > 1.5 and LeftRightSec[i] > 1.5:
			pbs.update_state(ServerSide.NR)
		scoreText1 = " %d, %d, 1" % (pbs.score1[0],pbs.score1[1]) if pbs.score1 else ""
		scoreText2 = " %d, %d, 2" % (pbs.score2[0],pbs.score2[1]) if pbs.score2 else ""

		if not scoreText2:
			scoreText = scoreText1
		elif not scoreText1:
			scoreText = scoreText2
		else:
			scoreText = scoreText1 + " or " + scoreText2
		if pbs.error:
			scoreText = ""
		print(convert(i,True) + " " + NFtext + LRtext + scoreText)
		#transcript += [["%s Serve" % (convert(i,True))]]
		transcript += [[convert(i,True)]]
		transcript += [[NFtext + LRtext + scoreText]]
		transcript += [[" "]]
	elif state == 1 and InServSec[i] <= 0.5:
		state = 0
	#print(state, i)

print(transcript)
with open('serveTranscript.csv', 'w') as csvfile: 
	# creating a csv writer object 
	csvwriter = csv.writer(csvfile, delimiter=";")
        
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
