#!/bin/bash -i

# best results with a video 1280x720

#set -e

# if docker pukes, do a newgrp docker in the terminal.

# $1 is the first arg, the full path name of a video to process.
# e.g. ./processIt.sh ~/Videos/senProf22.mp4

# get path
# get root of vidname
# get vidname

SECONDS=0 

inVid=$1
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
cp $inVid $filename/.
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
python ../extractFrame.py -v $vidname -f 1
# out testframe.png
python ../courtfind.py 
# out court.txt
#######################################################
#### create mask and per frame confidences that court is in frame.
# in: video. Also expects to find testframe.png and court.txt in cwd
python ../validate.py $vidname
# out: mask.png, target.csv, courtConf.csv
#######################################################
#### object detection and tracking for players, annotate video.
# in: video, courtConf.csv, mask.png, court.txt
echo "current env is" $CONDA_DEFAULT_ENV
python ../demo_track_pbcourt.py video -f ~/ByteTrack/exps/example/mot/yolox_x_mix_det.py -c ~/ByteTrack/models/bytetrack_x_mot17.pth.tar --path $vidname --fp16 --fuse --save_result -cc courtConf.csv  -mask mask.png -map court.txt
# out: byteTrackPbcourt.mp4, detectionsIds_peeps.txt rawDetectionsIds.txt
conda deactivate
#######################################################
pushd /home/fritz/test_tracknet2/TrackNetv2/3_in_3_out
#source ~/tracknet2/bin/activate
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
python ../show_trajectory_bounce.py byteTrackPbcourt.mp4 ${filename}_predict.csv courtConf.csv detectionsIds_peeps.txt 1

python ../postproc.py
deactivate
# out: byteTrackPbcourt_trajectory.mp4 with stats in video
# out: TODO ballInPlay, inServicePosition
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


