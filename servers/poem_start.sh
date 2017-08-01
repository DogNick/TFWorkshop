#!/bin/bash
PORT=9000
SERVICE=judgepoem
SCHEDULE=0
sh stop.sh $PORT 
nohup python -u Servers.py \
    --port=$PORT --procnum=4 \
    --service=$SERVICE --schedule=$SCHEDULE --deploy_root=deployments \
    --num_batch_threads=64 --batch_timeout_micros=100000 \
    --max_enqueued_batches=64 --max_batch_size=32 2>&1 > logs/server_$SERVICE"_"$SCHEDULE"_"$PORT".log" &
#
PORT=9001
SERVICE=judgepoem
SCHEDULE=1
sh stop.sh $PORT 
nohup python -u Servers.py \
    --port=$PORT --procnum=4 \
    --service=$SERVICE --schedule=$SCHEDULE --deploy_root=deployments \
    --num_batch_threads=64 --batch_timeout_micros=100000 \
    --max_enqueued_batches=64 --max_batch_size=32 2>&1 > logs/server_$SERVICE"_"$SCHEDULE"_"$PORT".log" &
PORT=9002
SERVICE=matchpoem
SCHEDULE=0
sh stop.sh $PORT 
nohup python -u Servers.py \
    --port=$PORT --procnum=4 \
    --service=$SERVICE --schedule=$SCHEDULE --deploy_root=deployments \
    --num_batch_threads=64 --batch_timeout_micros=100000 \
    --max_enqueued_batches=64 --max_batch_size=32 2>&1 > logs/server_$SERVICE"_"$SCHEDULE"_"$PORT".log" &

PORT=9003
SERVICE=matchpoem
SCHEDULE=1
sh stop.sh $PORT 
nohup python -u Servers.py \
    --port=$PORT --procnum=4 \
    --service=$SERVICE --schedule=$SCHEDULE --deploy_root=deployments \
    --num_batch_threads=64 --batch_timeout_micros=100000 \
    --max_enqueued_batches=64 --max_batch_size=32 2>&1 > logs/server_$SERVICE"_"$SCHEDULE"_"$PORT".log" &
