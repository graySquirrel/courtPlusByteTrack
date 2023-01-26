import cv2
import argparse
import csv
 
# Construct the argument parser and parse the arguments
arg_desc = '''\
           use -v to specify video name 
        '''
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                    description= arg_desc)
 
parser.add_argument("-v", "--video", metavar="VIDEO", help = "Path to your input video")
parser.add_argument("-d", "--detections", metavar="DETECTIONS", help = "detections file from ByteTrack")
args = vars(parser.parse_args())
 
 
if args["video"]:
    vidname = args["video"]
else:
    print("missing arg -v for video")
    quit() 
if args["detections"]:
    detectFile = args["detections"]
else:
    print("missing arg -d for detections")
    quit()

cap = cv2.VideoCapture(vidname)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps",fps)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length, "frames" )

# read in detections file
with open(detectFile) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	list1 = []
	frames = []
	thisFrame = 0
	tmpDetections = []
	for row in readCSV: # parse the csv to list elements
		list1.append(row)
	for i in range(0 , len(list1)):
		if int(list1[i][0]) != thisFrame: # 0th is frame num
			# close last frame
			frames.append(tmpDetections.copy())
			#print("frame",thisFrame,"len",len(frames[thisFrame]))
			#print(frames[thisFrame])
			if len(frames[thisFrame]) == 0:
				print("Frame",thisFrame,"has",len(frames[thisFrame]))
			# reset this frame
			tmpDetections = []
			# set new frame
			thisFrame = int(list1[i][0])
		# add this detection to current Frame
		tmpDetections.append([  int(float(list1[i][1])), # ID
								int(float(list1[i][2])), # X
								int(float(list1[i][3])), # Y
								int(float(list1[i][4])), # W
								int(float(list1[i][5])) ]) # H
	if tmpDetections != []:
		frames.append(tmpDetections.copy())

#for i in range(0,len(frames)):
	#tmpd = frames[i]
	#print(len(tmpd)," detections in frame ",i)
	#print(tmpd[0])
	#for j in range(0,len(frames[i])):
		#print(frames[i][j][0],frames[i][j][1],frames[i][j][2],frames[i][j][3],frames[i][j][4])

def isxyInBB(x, y, bb):
	if x > bb[1] and x < (bb[1] + bb[3]) and y > bb[2] and y < (bb[2] + bb[4]):
		return True, bb[0]
	else:
		return False, 0

############# make callback for mouseclicks

def click_capture(event, x, y, flags, param):
	global currentClickPt, myIds, IDselected, framenum
	if event == cv2.EVENT_LBUTTONDOWN:
		currentClickPt = [x, y]
		#print("click",currentClickPt)
		for bb in myIds:
			res, IDselected = isxyInBB(x, y, bb)
			if res:
				print("ID", IDselected, "selected on Frame", framenum)
				print("Now select 1:Serve, 2:Return, 3:Drop, 4:Drive, 5:Dink, 6:Volley, 7:Lob")
				return
		print("You clicked, but No bounding boxes found")

#############

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_capture)

currentClickPt = []
myIds = frames[0]
IDselected = 0
hitSelected = ""
fOrB = ""
framesNiDsNlabels = []

#############

# Read until video is completed
while (cap.isOpened()):
	# Capture frame-by-frame
	framenum = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) 
	#print("Frame", framenum) 

	ret, frame = cap.read()

	myIds = frames[framenum]
	#print(myIds)
	if ret == True:

		# Display the resulting frame
		cv2.imshow('Frame', frame)
		key = cv2.waitKey(0) & 0xFF
		#print(key)
		# Press Q on keyboard to  exit
		if key == ord('q'):
			break
		elif key == ord(','):
			if framenum > 1:
				cap.set(1, framenum-1)
			else:
				cap.set(1, 0)
			continue
		elif key == ord('.'):
			if framenum < length:
				cap.set(1, framenum+1)
			continue
		elif key == ord('<'): #This is less than not arrow
			if framenum > 10:
				cap.set(1, framenum-10)
			else:
				cap.set(1, 0)
			continue
		elif key == ord('>'): #greater than, not arrow
			if framenum < length-10:
				cap.set(1, framenum + 10)
			continue
		elif key == ord('1'):
			hitSelected = "Serve"
			print("1 (Serve) selected.  Now select F for forehand or B for backhand")
		elif key == ord('2'):
			hitSelected = "Return"
			print("2 (Return) selected.  Now select F for forehand or B for backhand")
		elif key == ord('3'):
			hitSelected = "Drop"
			print("3 (Drop) selected.  Now select F for forehand or B for backhand")
		elif key == ord('4'):
			hitSelected = "Drive"
			print("4 (Drive) selected.  Now select F for forehand or B for backhand")
		elif key == ord('5'):
			hitSelected = "Dink"
			print("5 (Dink) selected.  Now select F for forehand or B for backhand")
		elif key == ord('6'):
			hitSelected = "Volley"
			print("6 (Volley) selected.  Now select F for forehand or B for backhand")
		elif key == ord('7'):
			hitSelected = "Lob"
			print("7 (Lob) selected.  Now select F for forehand or B for backhand")
		elif key == ord('f'):
			fOrB = "Forehand"
			print("F (orehand) selected.  Press S to save annotation")
		elif key == ord('b'):
			fOrB = "Backhand"
			print("B (ackhand) selected.  Press S to save annotation")
		elif key == ord('s'):
			if IDselected == 0:
				print("please select a bounding box to annotate")
				continue
			elif hitSelected == "":
				print("please select a hit")
				continue
			elif fOrB == "":
				print("please select f(orehand) or b(ackhand)")
				continue
			print("Saving annotation on Frame:", framenum, "ID:", IDselected, "Label:", fOrB+hitSelected)
			framesNiDsNlabels.append([framenum, IDselected, fOrB+hitSelected])
			# clear out so we dont do it again
			IDselected = 0
			hitSelected = ""
			fOrB = ""
		cap.set(1, framenum)
	# Break the loop
	else:
		break

# save the annotation file
output_csv_path = vidname + "_annotation.csv"
with open(output_csv_path, 'w') as csvfile: 
	# creating a csv writer object 
	csvwriter = csv.writer(csvfile) 
	# writing the fields 
	csvwriter.writerows(framesNiDsNlabels) 

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
