#!/bin/bash
if [ $# -eq 1 ]
then
    TRAIN_ROOT=runtime
else
    TRAIN_ROOT=$1
fi

ps aux | grep tensorboard | grep -v grep | awk '{print $2}' | xargs -t -I {} kill -9 {}''
nohup tensorboard --logdir=/search/odin/Nick/GenerateWorkshop/runtime --debug 2>&1 > logs/board.log &
