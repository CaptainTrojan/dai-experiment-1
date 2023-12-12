#!/bin/bash

all_mp4=$(ls ../data/*.mp4 | wc -l)
det_mp4=$(ls ../data/*_det.mp4 2>/dev/null | wc -l)
expected_files=$(($all_mp4 - $det_mp4))
i=1

for file in ../data/*.mp4
do
    # Check if the processed file exists
    if [[ $file != *_det* ]] && [[ ! -e "${file}_det.mp4" ]]
    then
        echo "Processing $file ($i/$expected_files)"
        python object_detection.py -v $file
        i=$((i+1))
    fi
done
