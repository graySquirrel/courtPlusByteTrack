import cv2
import argparse
 
# Construct the argument parser and parse the arguments
arg_desc = '''\
           use -v to specify video name 
        '''
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                    description= arg_desc)
 
parser.add_argument("-v", "--video", metavar="VIDEO", help = "Path to your input video")
args = vars(parser.parse_args())
 
 
if args["video"]:
    vidname = args["video"]
else:
    print("missing arg -v for video")
    quit() 

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

cap = cv2.VideoCapture(vidname)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps",fps)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )


# When everything done, release the video capture object
cap.release()

