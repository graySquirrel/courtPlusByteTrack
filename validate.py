import numpy as np
import cv2
import csv
import sys
import os

# print(cv2.__version__)
# TODO argparse
#videofile = "../vids/ProSenior_00_05_27-40.mp4"
#videofile = "/home/vboxuser/share/ProSenior_00:05:27_20.mp4"
if len(sys.argv) < 2:
    sys.exit('Usage: python validate.py <input video>')

if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: input video %s was not found!' % sys.argv[1])

videofile = sys.argv[1]
framefile = "testframe.png"
datfile = "court.txt"
##
# validate.py will do the following:
# 1. get the perspective projection btw image coords and orthogonal coords of a pickleball court.
#       Then it will expand the court by 2.5 meters, and project into image coords.
#       Then it will create an image mask of these coords and write the mask out as 'mask.png'
# 2. read in a testframe.png (the frame used to create the court model), and output a file called "target.csv" which prints the max color in each of the 14 points on the court.
# 3. read the video, and for each frame, output a confidence of if the court is in the expected place.
#       Then write it out as 'courtConf.csv'
#########################################################################
# gets the max in a 11x11 window around each of the 14 court points.
win = 5


def getmaxes(img, points):
    maxes = np.array([])
    for point in points:
        xval = point[0]
        yval = point[1]
        #print("value at coord",point, "is", img[yval, xval])
        thismax = img[yval - win:yval + win, xval - win:xval + win].max()
        maxes = np.append(maxes, thismax)
        #print("max in neighborhood", thismax)
        # black it out to show it works
        #img[yval - win:yval + win, xval - win:xval + win] = [0, 0, 0]
    # print(maxes)
    return maxes
#########################################################################
# calculate a confidence score that the frame contains the court at the viewpoint
# where the court detection was calculated.
# has to be robust to some occlusion because players can cover points with
# arbitrary bits.  So, throw out the largest N points, then if the remaining
# errors are < T then it is a high conf, else not...


def getConf(img, targs, pts):
    maxes = getmaxes(img, pts)
    diffs = abs(targs - maxes)
    # print(targs)
    # print(maxes)
    # print(diffs)
    # throw out largest 4 diffs, then sum
    diffs[np.argmax(diffs)] = 0
    diffs[np.argmax(diffs)] = 0
    diffs[np.argmax(diffs)] = 0
    diffs[np.argmax(diffs)] = 0
    # print(diffs)
    # print(sum(diffs))
    # print(max(diffs))
    return round(1-max(diffs)/255, 2)  # a threshold of 0.85 makes sense.
#########################################################################


def drawCourtLines(img, points):
    imgout = cv2.line(img, points[0], points[1], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[1], points[2], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[2], points[3], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[3], points[0], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[4], points[7], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[5], points[6], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[8], points[10], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[9], points[11], [0, 0, 255], 3)
    imgout = cv2.line(imgout, points[12], points[13], [0, 0, 255], 3)
    return imgout

#########################################################################
# get the court points from file
# first 14 are in img coords, second 14 are in orthogonal coords


def getpoints(file):
    points = []
    intpoints = []
    orthopoints = []
    with open(file, mode='r') as file:
        csvFile = csv.reader(file)
        i = 0
        for line in csvFile:
            xval = float(line[0])
            yval = float(line[1])
            xvalint = int(xval)
            yvalint = int(yval)
            if i < 14:
                # print(i,line)
                points.append([xval, yval])
                intpoints.append([xvalint, yvalint])
            if i >= 14:
                orthopoints.append([xval, yval])
            i += 1
        # print(points)
        # print(orthopoints)
    return np.float32(points), \
        intpoints, \
        np.float32(orthopoints)
#########################################################################
# make projection, push outer bounds out 2.5m, project back into img space


def getoutbounds(orthpts, imgpts):
    resmatrix = cv2.getPerspectiveTransform(orthpts[:4], imgpts[:4])
    testpts = []
    limit = 2.0
    for pt in orthpts[:4]:
        x = pt[0]
        y = pt[1]
        if x == 0:
            x = -limit
        else:
            x += limit
        if y == 0:
            y = -limit
        else:
            y += limit
        testpts.append([x, y])
    # have to reshape the input to perspectiveTransform cray cray
    src = np.zeros((len(testpts), 1, 2))
    src[:, 0] = testpts
    outpts = cv2.perspectiveTransform(src, resmatrix)
    outpts = outpts[:, 0, :]
    print('bounds', outpts)
    return(outpts)
#########################################################################
# make an image mask of the points made from getoutbounds()


def getMask(framefile, shape):
    img = cv2.imread(framefile)
    themask = np.ones([img.shape[0], img.shape[1]], dtype='uint8')
    points = np.array(shape, dtype='int')
    print(points)
    toppoint = [[points[0][0], 0]]
    print(toppoint)
    bottompoint = [[points[3][0], 0]]
    print(bottompoint)
    points = np.append(toppoint, points, axis=0)
    points = np.append(points, bottompoint, axis=0)
    print(points)
    cv2.fillPoly(themask, pts=[points], color=([255]))
    return themask


#########################################################################
# get the points from the file
pts, intpts, orthpts = getpoints(datfile)
# get the court extent
boundspoints = getoutbounds(orthpts, pts)
# make a mask to encode it
# alter to don't mask the back part of the court
outmask = getMask(framefile, boundspoints)
cv2.imwrite("mask.png", outmask)
# get the initial max vals from each of 14 court points
img = cv2.imread(framefile)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
targmaxes = getmaxes(gray, intpts)
with open("target.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(targmaxes)
# get the per-frame metadata
framemetadata = []
cap = cv2.VideoCapture(videofile)
framenum = 0
while True:
    conf = 0
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    conf = getConf(gray, targmaxes, intpts)
    # print('conf',conf)
    # if conf > 0.85:
    #    frame = drawCourtLines(frame, intpts)
    framemetadata.append([framenum, conf])
    framenum += 1
    #cv2.imshow("the img", frame)
    # change waitkey to 0 if you want to wait for input to advance.
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    if framenum % 500 == 0:
        print(framenum)
cap.release()
cv2.destroyAllWindows()
# write metadata file
fields = ['FrameId', 'Conf']
with open("courtConf.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(framemetadata)
