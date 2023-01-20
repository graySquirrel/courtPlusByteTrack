import os
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import csv
import sys
import statistics

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=2,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

try:
	input_video_path = sys.argv[1]
	input_csv_path = sys.argv[2]
	court_conf_csv_path = sys.argv[3]
	player_loc_path = sys.argv[4]
	startFrame = int(sys.argv[5])
	if (not input_video_path) or (not input_csv_path):
		raise ''
except:
	print('usage: python3 show_trajectory.py <input_video_path> <input_csv_path> <court_csv_path> <player_loc_path> <start frame>')
	exit(1)
print(input_video_path)
print(input_csv_path)
print(court_conf_csv_path)
print(player_loc_path)
print(startFrame)

with open(court_conf_csv_path) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	conf = []
	list2 = []
	for row in readCSV:
		list2.append(row)
	for i in range(1, len(list2)):
		conf += [float(list2[i][1])]

with open(input_csv_path) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	frames = []
	x, y = [], []
	list1 = []
	visibility = []
	for row in readCSV:
		list1.append(row)
	for i in range(1 , len(list1)):
		frames += [int(list1[i][0])]
		visibility += [int(list1[i][1])]
		x += [int(float(list1[i][2]))]
		y += [int(float(list1[i][3]))]

with open(player_loc_path) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	peepXlocs = []
	peepYlocs = []
	list3 = []
	for row in readCSV:
		list3.append(row)
	for i in range(0, len(list3)): # no header row, so start at 1
		peepXlocs += [[float(list3[i][1]),float(list3[i][3]),float(list3[i][5]),float(list3[i][7])]]
		peepYlocs += [[float(list3[i][2]),float(list3[i][4]),float(list3[i][6]),float(list3[i][8])]]


output_video_path = input_video_path.split('.')[0] + "_trajectory.mp4"

q = queue.deque()
for i in range(0,8):
	q.appendleft(None)
print("len x",len(x))
print("len y",len(y))
print("len conf",len(conf))
print("len list3", len(list3))
print("len peepXlocs", len(peepXlocs))
print("len peepYlocs", len(peepYlocs))
endFrameCount = min(len(x),len(conf))
print("endcount",endFrameCount)
#print(peepYlocs[1:5])
#print(peepYlocs[len(peepYlocs)-5:len(peepYlocs)])
#quit()

#get video fps&video size
currentFrame= startFrame
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))

#video.set(1,currentFrame); 
#ret, img1 = video.read()
#write image to video
#output_video.write(img1)
#currentFrame +=1
#input must be float type
#img1 = img1.astype(np.float32)

#capture frame-by-frame
#video.set(1,currentFrame);
#ret, img = video.read()
#write image to video
#output_video.write(img)
#currentFrame +=1
#input must be float type
#img = img.astype(np.float32)

# init a counter that counts number of contiguous not-detections and number of contiguous detections
numcontigdetections = 0
numcontignondetections = 0
numframessincefull = 0

Vx = [0,0,0,0,0,0,0]
Vy = [0,0,0,0,0,0,0]
Vmag = [0,0,0,0,0,0,0]
Ax = [0,0,0,0,0,0]
Ay = [0,0,0,0,0,0]
FSLD = 0 # frames since long drive
inPlay = 0

# initialize output csv
outfields = ['frame','LongTermDetectRate', 'InPlay','InServicePos','nearFarServ','leftRightServ']
outrows = []
output_csv_path = input_video_path.split('.')[0] + "_metadata.csv"

while(True):
	if currentFrame >= endFrameCount:
		break
	#capture frame-by-frame
	if currentFrame % 500 == 0:
		print(currentFrame) 
	video.set(1,currentFrame+1); 
	ret, img = video.read()
		#if there dont have any frame in video, break
	if not ret: 
		break
	PIL_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
	PIL_image = Image.fromarray(PIL_image)
	#print(currentFrame, end=" ")
	if x[currentFrame] != 0 and y[currentFrame] != 0 and conf[currentFrame] > 0.85:
		q.appendleft([x[currentFrame],y[currentFrame]])
		q.pop()
		numcontignondetections = 0
		numcontigdetections += 1
	else:
		q.appendleft(None)
		q.pop()
		numcontigdetections = 0
		numcontignondetections += 1

## Need some outlier detection/rejection on ball positions...
	for i in range(0,7):
		#print(q[i])
		if q[i] != None and q[i+1] != None:
			Vx[i] = q[i+1][0] - q[i][0]
			Vy[i] = q[i+1][1] - q[i][1]
			if abs(Vx[i]) > 100 or abs(Vy[i]) > 100: # outlier
				Vx[i] = 0
				Vy[i] = 0
			Vtmp = np.array([Vx[i],Vy[i]])
			Vmag[i] = np.sqrt(Vtmp.dot(Vtmp))
		else:
			Vx[i] = 0
			Vy[i] = 0

	queuefull = True
	for i in range(0,8):
		if q[i] is not None:
			draw_x = q[i][0]
			draw_y = q[i][1]
			bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
			draw = ImageDraw.Draw(PIL_image)
			draw.ellipse(bbox, outline ='yellow')
			del draw
		else:
			queuefull = False

	meanVmag = 0
	stdV = 0
	if queuefull:
		numframessincefull = 0
		meanVmag = np.array(Vmag).mean()
		stdV = statistics.pstdev(Vmag)
		#print(meanVmag,stdV)
		FSLD = 0 if meanVmag > (stdV * 5) else FSLD + 1
	else:
		numframessincefull += 1
		FSLD += 1

	firstFrame = currentFrame - 100 if currentFrame > 100 else 0
	longTermDetectRate = 0
	for i in range(firstFrame,currentFrame):
		if visibility[i] == 1 and conf[i] > 0.85:
			longTermDetectRate += 1
	longTermDetectRate = longTermDetectRate / (currentFrame - firstFrame)


	opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

	thisYs = peepYlocs[currentFrame]
	thisXs = peepXlocs[currentFrame]
	numAtSL =      len([i for i in thisYs if (i<1.0 or i>12.41) and i != 0.00]) # court dims 0<Y<13.41m
	numNear =      len([i for i in thisYs if (i>12.41)])
	numNearSide =  len([i for i in thisYs if (i<6.705)]) # the net line is 6.705, 13.41/2
	numFarSide =   len([i for i in thisYs if (i>6.705)]) # remember 0,0 is back left.
	numFar  =      len([i for i in thisYs if (i<1.0)])
	numInXbounds = len([i for i in thisXs if i>-1.0 and i<7.1   and i != 0.00]) # court dims 0<X<6.1m
	numNotAtSL =   len([i for i in thisYs if (i>1.0 and i<12.41) and i != 0.00]) # 
	servicePos = 1 if (numAtSL == 3 and numInXbounds == 4 and numNotAtSL == 1 and (numNearSide==2 and numFarSide==2)) and conf[currentFrame] > 0.85 else 0
	inPlay = 1 if (longTermDetectRate > 0.5 or FSLD < 100 or (numNotAtSL > 1 and numInXbounds == 4)) and conf[currentFrame] > 0.85 else 0
	nearFar = 0 # 0 is none
	leftRightServer = 0 # the Server is on left (1) or right (2) side as they look at the court
	if (servicePos and numFar == 2 and numNear == 1):
		nearFar = 1 # 1 is far
		for a in range(0,4): # there is only one numNear...
			if thisYs[a] > 12.41:
				if thisXs[a] < 3.05: # The Near person is on Left, so far server is on their left
					leftRightServer = 1
				else:
					leftRightServer = 2
				break
	if (servicePos and numFar == 1 and numNear == 2):
		nearFar = 2 # 2 is near
		for a in range(0,4): # there is only one numFar...
			if thisYs[a] < 1.0:
				if thisXs[a] < 3.05: # The Far person is on their Right, so near server is on their right
					leftRightServer = 2
				else:
					leftRightServer = 1
				break

	outrows += [[currentFrame, longTermDetectRate, inPlay, servicePos, nearFar, leftRightServer]]

	meanV = "meanVm: %.2f" % meanVmag
	stdV = "stdVm: %.2f" % stdV
	D = "D: %d" % (numcontigdetections)
	N = "N: %d" % (numcontignondetections)
	F = "Full: %d" % (queuefull)
	NFSF = "NFSF: %d" % (numframessincefull)
	CF = "Frame: %d" % (currentFrame)
	LTDR = "LTDR: %.2f" % (longTermDetectRate)
	FSLDstr = "FSLD: %d" % FSLD
	inPlaystr = "in play: %d" % inPlay
	inSvcPos = "Serving: %d" % servicePos
	w = int(output_width * 0.8)
	draw_text(opencvImage, D, pos=(w,50))
	draw_text(opencvImage, N, pos=(w,100))
	draw_text(opencvImage, F, pos=(w,150))
	draw_text(opencvImage, NFSF, pos=(w,200))
	draw_text(opencvImage, CF, pos=(w,250))
	draw_text(opencvImage, meanV, pos=(w,300))
	draw_text(opencvImage, stdV, pos=(w,350))
	draw_text(opencvImage, LTDR, pos=(w,400))
	draw_text(opencvImage, FSLDstr, pos=(w,450))
	draw_text(opencvImage, inPlaystr, pos=(w,500))
	draw_text(opencvImage, inSvcPos, pos=(w,550))
	#write image to output_video
	if False:
		cv2.imshow("img", opencvImage)
		ch = cv2.waitKey(0)
		if ch == 27 or ch == ord("q") or ch == ord("Q"):
			break
	output_video.write(opencvImage)

	#next frame
	currentFrame += 1

video.release()
output_video.release()

with open(output_csv_path, 'w') as csvfile: 
	# creating a csv writer object 
	csvwriter = csv.writer(csvfile) 
        
	# writing the fields 
	csvwriter.writerow(outfields) 
        
	# writing the data rows 
	csvwriter.writerows(outrows)

print("finish")

