#!/bin/bash
PORT=9010
SERVICE=chaten
SCHEDULE=0
sh stop.sh $PORT 
#TF_CPP_MIN_VLOG_LEVEL=1 
nohup python -u Servers.py \
    --port=$PORT --procnum=4 \
    --service=$SERVICE --schedule=$SCHEDULE --deploy_root=deployments \
    --num_batch_threads=64 --batch_timeout_micros=80000 \
    --max_enqueued_batches=10000 --max_batch_size=32 2>&1 > logs/server_$SERVICE"_"$SCHEDULE"_"$PORT".log" &
