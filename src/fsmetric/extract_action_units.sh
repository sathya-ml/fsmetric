#!/bin/bash

# This script extracts action units from a video file using the OpenFace tool.
# It runs the FaceLandmarkVidMulti tool in a Docker container and outputs a CSV file
# containing the extracted action units.

VID_PATH=$1
OUTPUTS_DIR=$2

FULL_FN="${VID_PATH##*/}"
TAG="${FULL_FN%.*}"

# Create a temporary directory for output files
mkdir $OUTPUTS_DIR/temp

# Run the OpenFace FaceLandmarkVidMulti tool using Docker
docker run --rm -it \
  --entrypoint="/home/openface-build/build/bin/FaceLandmarkVidMulti" \
  -v $VID_PATH:/root/videos/$TAG.file \
  -v $OUTPUTS_DIR/temp:/root/out/ \
  algebr/openface:latest \
  -f /root/videos/$TAG.file -out_dir /root/out/ -q

# Move the resulting CSV file to the output directory
mv $OUTPUTS_DIR/temp/$TAG.csv $OUTPUTS_DIR/

# Remove the temporary directory
rm -rf $OUTPUTS_DIR/temp
