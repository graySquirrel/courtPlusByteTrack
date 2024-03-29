#!/bin/bash -i

# processIt_justBBs.sh will dewarp, tag bounding boxes. Skips validate.py and 

#set -e

# if docker pukes, do a newgrp docker in the terminal.

# $1 is the first arg, the full path name of a video to process.
# $2 is second arg, a boolean (0, 1) if 1 will skip asking user to input court corners
# $3 is third arg, a boolean (0, 1) if 1 will dewarp the wide angleness of the video. (you have to know)

# e.g. ./processIt.sh ~/Videos/senProf22.mp4

# get path
# get root of vidname
# get vidname

SECONDS=0 

inVid=$1
ignoreCourt=0
dewarpVideo=0
ARGC=$#
#echo $ARGC
if [ $ARGC -eq 3 ];
then
	if [ $2 -eq 1 ];
	then
	ignoreCourt=1
	fi
	if [ $3 -eq 1 ];
	then
	dewarpVideo=1
	fi
else
	echo "Usage: ./processIt.sh <video name> <skip court inputs 0,1> <dewarp wide angle 0,1>"
	exit
fi
#echo $ignoreCourt
# if $1 stays defined then conda craps out because it must be looking for that to do something.
set -- ""
vidname="${inVid##*/}"
echo $vidname
vidpath="${inVid%/*}"
echo $vidpath

extension="${vidname##*.}"
filename="${vidname%.*}"
echo $filename
echo $extension

thisdir=`pwd`

mkdir -p $filename
#cp $inVid $filename/.
#######################################################
# None of the court detect stuff is robust enough. doing manually ##########
#######################################################
#### extract frame for court finding
#cd $filename
#python ../extractFrame.py -v $vidname -f 1
#cd ..
#### detect court
# in: video
#docker run --rm -v ~/courtPlusByteTrack/$filename:/pickleball-court-detection/build/data detect:latest ./data/extractedFrame.jpg ./data
# out: court.txt, testframe.png, testframeWithLines.png
#######################################################
cd $filename
# activate bytetrack conda env
#conda activate bytetrack
# this is aliased in .bashrc
deactivate
my_bytetrack 
echo "current env is" $CONDA_DEFAULT_ENV

#######################################################
# Instead, we'll do manually ##########################
######### do manual court finding in bytetrack env ####
if [ $ignoreCourt -ne 1 ];
then
	echo "extracting frame and running courtfind"
	# out testframe.png
	if [ $dewarpVideo -eq 1 ];
	then
	echo "extracting frame, dewarping and running courtfind"
		python ../extractFrame.py -v $vidname -f 1 -d
	else
	echo "extracting frame and running courtfind"
		python ../extractFrame.py -v $vidname -f 1
	fi
	python ../courtfind.py
else
	echo "using existing court info"
fi
# out court.txt
#######################################################
# dewarp wide angle lens (assumes Hero9)
#######################################################
if [ $dewarpVideo -eq 1 ];
then
	echo "dewarping video"
	#python ../lensfunCorrect.py -v $vidname
	vidname=${vidname}_undistorted.mp4
else
	echo "using existing court info"
fi
echo "video name is " $vidname
#######################################################
#### create mask and per frame confidences that court is in frame.
# in: video. Also expects to find testframe.png and court.txt in cwd
python ../validate.py -v $vidname
# out: mask.png, target.csv, courtConf.csv
#######################################################
#### object detection and tracking for players, annotate video.
# in: video, courtConf.csv, mask.png, court.txt
echo "current env is" $CONDA_DEFAULT_ENV
#python ../demo_track_pbcourt.py video -f ~/ByteTrack/exps/example/mot/yolox_m_mix_det.py -c ~/ByteTrack/models/bytetrack_m_mot17.pth.tar --path $vidname --fp16 --fuse --save_result -mask mask.png -map court.txt
# out: byteTrackPbcourt.mp4, detectionsIds_peeps.txt rawDetectionsIds.txt
conda deactivate
#######################################################
pushd /home/fritz/test_tracknet2/TrackNetv2/3_in_3_out
source ~/tracknet2/bin/activate
# aliased in .bashrc
my_tracknet2
#######################################################
#### tracknet2
# in: video, 
python3 predict3.py --video_name=$thisdir/$filename/$vidname --load_weights="model906_30"
# out: predict.csv with ball position predictions and conf.
#######################################################
popd
#######################################################
#### show ball trajectory, calculate point metadata
# in: byteTrackPbcourt.mp4, predict.csv, courtConf.csv, detectionsIds_peeps.txt
# python ../show_trajectory_bounce.py
python ../show_trajectory_bounce.py byteTrackPbcourt.mp4 ${filename}_predict.csv courtConf.csv detectionsIds_peeps.txt 0 #startframe

python ../postproc.py
deactivate
# out: byteTrackPbcourt_trajectory.mp4 with stats in video

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


