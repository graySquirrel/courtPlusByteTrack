import cv2
import numpy as np
import csv

# Do the points top left, bottom left, bottom right, top right.

imgOrig = cv2.imread("testframe.png")
img = imgOrig.copy()


dimensions = img.shape
print(dimensions)

windowname = "courtfind"
thepoints = []


def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print("hello " + str(x) + " " + str(y))
        cv2.circle(img, (x, y), 100, (0, 255, 0), -1)


def draw_poly(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #img = imgOrig.copy()
        print("adding point")
        thepoints.append([x, y])
        print(thepoints)
        for p in thepoints:
            cv2.circle(img, p, 10, (0, 255, 0), -1)
        pts = np.array(thepoints, np.int32)
        pts = pts.reshape((-1, 1, 2))
        print(pts)
        if len(thepoints) < 4:
            isClosed = False
        else:
            isClosed = True
        print("isclosed", isClosed)
        cv2.polylines(img, [pts], isClosed, (0, 255, 0), 5)


def sortNsave():
    # sort the list so it goes 0,0 0,long wide,long wide,0
    # write out the court.txt file
    rows = [[thepoints[0][0], thepoints[0][1]],
            [thepoints[1][0], thepoints[1][1]],
            [thepoints[2][0], thepoints[2][1]],
            [thepoints[3][0], thepoints[3][1]],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
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
