#!/bin/bash

all_mp4=$(ls ../data/*.mp4 | wc -l)
# det_mp4=$(ls ../data/*_det.mp4 2>/dev/null | wc -l)
expected_files=$(($all_mp4))
i=1

for file in ../data/*.mp4
do
    echo "Processing $file ($i/$expected_files)"
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of csv=s=,:p=0 $file
    i=$((i+1))
done
