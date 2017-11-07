#!/bin/bash
#PORT=9000
#SERVICE=generate
#SCHEDULE=4
#sh stop.sh $PORT 
#nohup python -u Servers.py \
#    --port=$PORT --procnum=4 \
#    --service=$SERVICE --schedule=$SCHEDULE --deploy_root=deployments \
#    --num_batch_threads=64 --batch_timeout_micros=80000 \
#    --max_enqueued_batches=10000 --max_batch_size=32 2>&1 > logs/server_$SERVICE"_"$SCHEDULE"_"$PORT".log" &

PORT=9001
SERVICE=generate
SCHEDULE=cvae_posterior_cn
sh stop.sh $PORT 
nohup python -u Servers.py \
    --port=$PORT --procnum=4 \
    --service=$SERVICE --schedule=$SCHEDULE --deploy_root=deployments \
    --num_batch_threads=64 --batch_timeout_micros=80000 \
    --max_enqueued_batches=10000 --max_batch_size=32 2>&1 > logs/server_$SERVICE"_"$SCHEDULE"_"$PORT".log" &
