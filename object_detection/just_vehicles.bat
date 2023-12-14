@echo off
setlocal enabledelayedexpansion
set "path_to_files=D:\"

for /F %%i in (just_vehicles.txt) do (
    set "filename=%%i"
    python object_detection.py -v "!path_to_files!\!filename!"
)
