#!/bin/bash

for file in ../data/*.mp4
do
    # Check if the processed file exists
    if [[ $file == *_det* ]]
    then
        echo "$file"
        rm $file
    fi
done
