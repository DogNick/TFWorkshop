#!/bin/bash

#########################################################################
# Nick use this to make final preparation of data for training and send it 
# to dist machine
####

#### configurable ####
VALID_NUM=100000
#####

if [ $# -ne 2 ]
then
    echo "Usage: sh "$0" data dist_dir"
    exit
fi

DATA=$1
PREPARE_DIR=$2
SHUFFLED=$DATA".shuffled"

DATA_LINE_NUM=`wc -l $DATA | awk '{print $1}'`
echo "Data line: "$DATA_LINE_NUM
TRAIN_NUM=$(($DATA_LINE_NUM - $VALID_NUM))
echo "Split into Train: "$TRAIN_NUM", Valid: "$VALID_NUM

# split data into train.data and valid.data
echo "Shuffling..."
shuf $DATA > $SHUFFLED
echo "Spliting..."
awk -v data_dir=$PREPARE_DIR -v train_num=$TRAIN_NUM '{if(NR>train_num){print $0 > data_dir"/valid.data"}else{print $0 > data_dir"/train.data"}}' $SHUFFLED


