import cv2
import numpy as np
import csv
import argparse
import sys
from lensfunCorrect import getUndistortedCoordinates

# Construct the argument parser and parse the arguments
arg_desc = '''\
           use -d to dewarp the test image 
        '''
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                    description= arg_desc)
 
parser.add_argument("-d", "--dewarp", action='store_true')
args = vars(parser.parse_args())
#print(args)
dewarp = False
if args["dewarp"]:
    dewarp =  True
    print("dewarp",dewarp)

# Do the points top left, bottom left, bottom right, top right.

imgOrig = cv2.imread("testframe.png")
img = imgOrig.copy()


dimensions = img.shape
#print(dimensions)

windowname = "courtfind"
thepoints = []


def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print("hello " + str(x) + " " + str(y))
        cv2.circle(img, (x, y), 100, (0, 255, 0), -1)


def draw_poly(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #img = imgOrig.copy()
        #print("adding point")
        thepoints.append([x, y])
        if len(thepoints) == 1:
            print("click BOTTOM LEFT")
        elif len(thepoints) == 2:
            print("click BOTTOM RIGHT")
        elif len(thepoints) == 3:
            print("click TOP RIGHT")
        elif len(thepoints) == 4:
            print("click 's' to save and show the court lines, then 'q' to quit")
        #print(thepoints)
        for p in thepoints:
            cv2.circle(img, p, 10, (0, 255, 0), -1)
        pts = np.array(thepoints, np.int32)
        pts = pts.reshape((-1, 1, 2))
        #print(pts)
        if len(thepoints) < 4:
            isClosed = False
        else:
            isClosed = True
        #print("isclosed", isClosed)
        cv2.polylines(img, [pts], isClosed, (0, 255, 0), 5)

def drawCourtLines(img, points):
	img = cv2.line(img, points[0], points[1], [0, 0, 255], 3)
	img = cv2.line(img, points[1], points[2], [0, 0, 255], 3)
	img = cv2.line(img, points[2], points[3], [0, 0, 255], 3)
	img = cv2.line(img, points[3], points[0], [0, 0, 255], 3)
	img = cv2.line(img, points[4], points[7], [0, 0, 255], 3)
	img = cv2.line(img, points[5], points[6], [0, 0, 255], 3)
	img = cv2.line(img, points[8], points[10], [0, 0, 255], 3)
	img = cv2.line(img, points[9], points[11], [0, 0, 255], 3)
	img = cv2.line(img, points[12], points[13], [0, 0, 255], 3)


def sortNsave():
    # make projection
	orthopoints = [[0, 0],
            [0, 13.41],
            [6.1, 13.41],
            [6.1, 0],
            [0, 4.57],
            [0, 8.84],
            [6.1, 8.84],
            [6.1, 4.57],
            [3.05, 0],
            [3.05, 13.41],
            [3.05, 4.57],
            [3.05, 8.84],
            [0, 6.71],
            [6.1, 6.71]]
	#print(thepoints[:4])
	#print(orthopoints[:4])
	xform = cv2.getPerspectiveTransform(np.float32(orthopoints[:4]), np.float32(thepoints[:4]))
	src = np.zeros((len(orthopoints), 1, 2))
	src[:, 0] = orthopoints
	#print("src",src)
	outpt = cv2.perspectiveTransform(src, xform)
	# write out the court.txt file
	#print(outpt)
	outpt = np.round(outpt, 0).astype(int)
	outpt = outpt[:,0,:]
	outpt = outpt.tolist()
	#print(outpt)
	#print(orthopoints)
	rows = outpt + orthopoints
	drawCourtLines(img, outpt)

	filename = "court.txt"
	with open(filename, 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerows(rows)


cv2.namedWindow(winname=windowname)
#cv2.setMouseCallback(windowname, draw_circle)
cv2.setMouseCallback(windowname, draw_poly)
height, width = img.shape[0], img.shape[1]
if dewarp:
    un = getUndistortedCoordinates(width, height)
    imgOrig = cv2.remap(imgOrig, un, None, cv2.INTER_LANCZOS4)
img = imgOrig.copy()

print("click TOP LEFT")
while True:
    cv2.imshow(windowname, img)

    k = cv2.waitKey(10)
    if k & 0xFF == ord('q'):
        break
    if k & 0xFF == ord('r'):
        print("resetting points")
        thepoints = []
        img = imgOrig.copy()
    if k & 0xFF == ord('s'):
        if len(thepoints) != 4:
            print("the number of points needs to be 4, it is ", len(thepoints))
        else:
            sortNsave()
            #break # wait for user

cv2.destroyAllWindows()
