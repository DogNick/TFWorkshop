#!/bin/bash
if [ $# -ne 0 ]
then
    PORT=$1
    echo "[Kill Port $PORT]"
    ps aux | grep Servers.py | grep -v "grep" | grep "port=$PORT" | awk '{print $2}' | xargs -t -I {} kill {}
else
    echo "[Kill All]"
    ps aux | grep Servers.py | grep -v "grep" | awk '{print $2}' | xargs -t -I {} kill {}
fi
