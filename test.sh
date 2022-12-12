#!/bin/bash -i
#source /home/fritz/.bashrc
#alias
#date 
#SECONDS=0
inVid=$1
set -- ""
echo $inVid
#vidname="${inVid##*/}"
#echo $vidname
#vidpath="${inVid%/*}"
#echo $vidpath

#extension="${vidname##*.}"
#filename="${vidname%.*}"
#echo $filename
#echo $extension

#thisdir=`pwd`
#echo $thisdir

#mkdir -p $filename
#cp $inVid $filename/.

echo $SECONDS
#cd $filename
# activate bytetrack conda env
#conda activate bytetrack
# this is aliased in .bashrc
#which conda
my_bytetrack 
return
