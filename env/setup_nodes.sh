#!/bin/bash

while read -u10 line 
do
    if [[ ${line:0:1} = "#" ]]
    then
        continue
    fi
    #nohup ./env_install.expect ${line} &
    ./install.expect ${line}
done  10< nodes 
