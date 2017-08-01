#!/bin/bash
pub_key=/search/odin/Nick/.ssh/id_rsa.pub

cat nodes | while read line 
do
    if [[ ${line:0:1} = "#" ]]
    then
        continue
    fi
    ./auth.expect $pub_key ${line}
done
