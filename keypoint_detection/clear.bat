@echo off
setlocal enabledelayedexpansion
set "path_to_files=D:\"

for /R "%path_to_files%" %%G in (pexels_walk*det*.*) do (
    del "%%G"
)
