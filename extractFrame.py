import cv2
import argparse
 
# Construct the argument parser and parse the arguments
arg_desc = '''\
           use -v to specify video name use -f to specify frame number
        '''
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                    description= arg_desc)
 
parser.add_argument("-v", "--video", metavar="VIDEO", help = "Path to your input video")
parser.add_argument("-f", "--frame_num", metavar="FRAMENUM", help = "The frame you want to extract")
args = vars(parser.parse_args())
 
 
if args["video"]:
    vidname = args["video"]
else:
    print("missing arg -v for video")
    quit() 
if args["frame_num"]:
	frame_num = int(args["frame_num"])
else:
	print("missing arg -f for frame number, using 0")
	frame_num = 0
outframe = "extractedFrame.jpg"

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

cap = cv2.VideoCapture(vidname)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)

# Check if camera opened successfully
if (cap.isOpened()== False):
	print("Error opening video stream or file")

ret, frame = cap.read()
if ret == True:
	cv2.imwrite(outframe, frame)

# When everything done, release the video capture object
cap.release()

