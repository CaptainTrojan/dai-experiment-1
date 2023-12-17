@echo off
setlocal enabledelayedexpansion
set "path_to_files=D:\\"
set "temp_file=%temp%\\file_list.txt"

REM Enumerate all fitting files and write their paths to the temporary file
for /R "%path_to_files%" %%G in (pexels_walk*.mp4) do (
    echo %%G >> "%temp_file%"
)

REM Loop over the lines in the temporary file and process each file
for /F "usebackq delims=" %%H in ("%temp_file%") do (
    python keypoint_detection.py -v "%%H"
)

REM Delete the temporary file
del "%temp_file%"
