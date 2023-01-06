import cv2
import numpy as np
import csv

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
	filename = "court.txt"
	with open(filename, 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerows(rows)


cv2.namedWindow(winname=windowname)
#cv2.setMouseCallback(windowname, draw_circle)
cv2.setMouseCallback(windowname, draw_poly)


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
            break

cv2.destroyAllWindows()
