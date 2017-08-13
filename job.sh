#!/bin/bash 
# Params for this script to work properly 

TRAIN_ROOT=runtime
RUNTIME_NAME=
JOB_TYPE=

INDEX=
NUM_TO_AGG=
GPU=

function runtime_checkpoint_info()
{
    train_dir=$1
    global_step=`tail -300 $train_dir/train.log | tac |  awk '/global/{ print }' | awk '{if(NR==1) print $3}'` 
    dev=`tail -100 $train_dir/train.log | tac | awk '
          /eval/{
              a[$3]++;
              if(a[$3]==1){b[$3]=$5}
          }END{
              len = length(b); 
              for(i=0; i < len; ++i)
              {
                  print b[i]
              }
          }' | tr "\n" "/"`
    echo $global_step"-->"$dev
}

function update_runtime_status()
{
    train_dir=$1
    ALIVE=`ps aux | awk '{print $2}' | tr "\n" " "` 
    if [ ! -f $train_dir/status ] 
    then
        continue;
    fi
    cat $train_dir/status | awk -v alive="$ALIVE" -F"\t" '
         BEGIN{
             a[-1] = 0;
             b[-1] = 0;
             c[-1] = 0;
             pids[-1] = 1;
             split(alive,c," ");
             for(each in c)
                 pids[c[each]] = 1;
         }{
             if(!($2 in pids))
             {
                 $2 = "----";
             }
             if (!($1 in a) || (a[$1] == "----"))
             {
                 a[$1] = $2;
                 b[$1] = $1"\t"$2"\t"$3"\t"$4;
             }
         }END{
            for (each in b)
            {
                if (each != -1)
                    print b[each] 
            }
         }' > $train_dir/.status.tmp
    cp $train_dir/.status.tmp $train_dir/status
    return
}

    
if [ $# -lt 1 ]
then
    echo ""
    echo ""
    echo "****************************** Uasage *************************************"
    echo "Create a job as ps(id), worker(id) or single " 
    echo "    sh "$0" create [runtime_name] [job_type(ps/worker/single)] [job_idx] [gpu]"
    echo ""
    echo "Stop all"
    echo "    sh "$0" stop"
    echo "Stop a process by runtime name"
    echo "    sh "$0" stop [runtime_name]"
    echo ""
    echo "See all runing:"
    echo "    sh "$0" see"
    echo "See model log:"
    echo "    sh "$0" see [model_name]"
    echo "See model's job log:"
    echo "    sh "$0" see [model_name] [job_type]"
    echo "************************************************************"
    echo ""
    exit
fi

if [ "$1"x = "create"x ]
then
    RUNTIME_NAME=$2
    JOB_TYPE=$3
    INDEX=$4
    GPU=$5

    TRAIN_DIR=$TRAIN_ROOT/$RUNTIME_NAME

    if [[ "$JOB_TYPE"x = "ps"x ]] || [[ "$JOB_TYPE"x = "worker"x ]] || [[ "$JOB_TYPE"x = "single"x ]]
    then
        touch $TRAIN_ROOT/run.pid
        EXISTS=`awk -v job_name=$JOB_TYPE -v idx=$INDEX '{if($1==job_name"_"idx)print}' $TRAIN_ROOT/run.pid` 
        if [ ! -z $EXISTS ] 
        then
           echo -e "\nAready Runing: \n"$EXISTS"\n" 
           exit
        fi
        
        mkdir -p $TRAIN_DIR
        touch $TRAIN_DIR/"train_"$JOB_TYPE"_"$INDEX".log"
        touch $TRAIN_DIR/"graph_"$JOB_TYPE"_"$INDEX".log"
        touch $TRAIN_DIR/"std_"$JOB_TYPE"_"$INDEX".log"
        CUDA_VISIBLE_DEVICES=$GPU nohup python -u workshop.py --cmd=train --job_type=$JOB_TYPE --task_id=$INDEX --gpu=$GPU --train_root=$TRAIN_ROOT --conf_name=$RUNTIME_NAME 2>&1 > $TRAIN_DIR/"std_"$JOB_TYPE"_"$INDEX".log" & 

        TIME=`date "+%Y-%m-%d %T"`
        echo $! | xargs -I {} echo -e $JOB_TYPE"_"$INDEX"\t"{}"\t"$DATA_DIR"\t"$TIME  >> $TRAIN_DIR/status
        #$TRAIN_DIR/log_$JOB_TYPE"_"$INDEX &
        #TIME=`date "+%Y-%m-%d %T"`
        #vocab_size=`head -1 $DATA_DIR/statistics | awk '{print $3}'`
        # echo $! | xargs -I {} echo -e $JOB_TYPE"\t"{}"\t"$DATA_DIR"("$vocab_size")\tgpu="$GPU"\t"$TIME >> $TRAIN_DIR/status
    else
        echo "Create a job as ps(id), worker(id) or single " 
        echo "    sh "$0" create [model_name] [job_type(ps/worker/single)] [job_idx] [data_dir] [gpu]"
        exit
    fi
elif [ "$1"x = "stop"x ]
then
    if [[ $# -ne 1 ]] && [[ $# -ne 2 ]] && [[ $# -ne 3 ]]
    then
        echo "Stop all"
        echo "    sh "$0" stop"
        echo "Stop a process by model name"
        echo "    sh "$0" stop [model_name]"
        echo ""
        exit
    fi
    if [ $# -eq 1 ]
    then
        runtimes=`ls -l $TRAIN_ROOT | awk '/^d/{print $9}' | tr "\n" " "`
        for each in $runtimes
        do
            cat $TRAIN_ROOT/$each/status | awk -F"\t" '{if($2 != "----")print $2}' | xargs -t -I {} kill -9 {}
        done
        sleep 0.2
        for each in $runtimes
        do
            update_runtime_status $TRAIN_ROOT/$each
        done

    elif [ $# -eq 2 ]
    then
        RUNTIME_NAME=$2
        awk -v runtime_name=$RUNTIME_NAME '{print $2}' $TRAIN_ROOT/$RUNTIME_NAME/status | xargs -t -I {} kill -9 {}
        update_runtime_status $TRAIN_ROOT/$RUNTIME_NAME
    else
        echo ""
        echo "Stop all:"
        echo "    sh "$0" stop"
        echo "Stop a process by model name:"
        echo "    sh "$0" stop [model_name]"
        echo ""
        exit
    fi
elif [ "$1"x = "see"x ]
then
    if [ $# -eq 1 ]
    then
        clear
        line=""
        for i in `seq 0 150`
        do
           line=$line"="
        done

        echo "  WORKFLOW"
        echo $line 
        runtimes=`ls -l $TRAIN_ROOT | awk '/^d/{print $9}' | tr "\n" " "`
        for each in $runtimes
        do
            update_runtime_status $TRAIN_ROOT/$each
            update_runtime_status $TRAIN_ROOT/$each
            #checkpoint_info=$(runtime_checkpoint_info $TRAIN_ROOT/$each)
            #cat $TRAIN_ROOT/$each/status | awk -v name=$each -v info=$checkpoint_info -F"\t" '
            #             {printf("  %-40s%-10s%-8s%-40s%-8s%-25s%-20s\n",name,$1,$2,$3,$4,$5,info)}'

            sort -k1 -k2 $TRAIN_ROOT/$each/status | awk -v name=$each -F"\t" '{printf("  %-40s%-10s%-8s%-40s\n",name,$1,$2,$4)}'
            #cat $TRAIN_ROOT/$each/status |
        done    
        line=""
        for i in `seq 0 150`
        do
           line=$line"="
        done
        echo $line 
        echo ""
    elif [ $# -eq 2 ]
    then
        TRAIN_DIR=$TRAIN_ROOT/$2

        GENERATE_LOGS=`awk -v traindir=$TRAIN_DIR 'BEGIN{a="";}{if($2!="----"){a=a" -wh 17 -cS Nick "traindir"/std_"$1".log -wh 17 -cS Nick "traindir"/graph_"$1".log -cS Nick "traindir"/train_"$1".log"}}END{print a}' $TRAIN_DIR/"status"`
        echo $GENERATE_LOGS
        NUM=`awk '{if($2!="----"){print;}}' $TRAIN_DIR/status | wc -l`
        echo $NUM
        if [ $NUM -eq 1 ]
        then
            multitail $GENERATE_LOGS 
        else
            multitail -s $NUM $GENERATE_LOGS 
        fi
#        echo "multitail -f -wh 13 -cS Nick $TRAIN_DIR/std.log -wh 30 -cS Nick $TRAIN_DIR/graph.log -cS Nick $TRAIN_DIR/train.log"
    elif [ $# -eq 3 ]
    then
	# TODO..
        TRAIN_DIR=$TRAIN_ROOT/$2
        JOB_TYPE=$3
        tail -f $TRAIN_DIR/log_$3

    else
        echo ""
        echo "See all runing:"
        echo "    sh "$0" see"
        echo "See model log:"
        echo "    sh "$0" see [model_name]"
        echo "See model's job log:"
        echo "    sh "$0" see [model_name] [job_type]"
    fi
elif [ "$1"x = "export_schedule"x ]
then
    SERVICE=$2
    SCHEDULE=$3
    echo "exporting schedule $SERVICE $SCHEDULE ..."
    python workshop.py --cmd=export --service=$SERVICE --schedule=$SCHEDULE --train_root=$TRAIN_ROOT
else:
elif [ "$1"x = "dummytrain"x ]
then
    RUNTIME_NAME=$2
    GPU=$3
    echo "Dummy Train for $RUNTIME_NAME on $GPU ..."
    python workshop.py --cmd=dummytrain --conf_name=$RUNTIME_NAME --gpu=$GPU --train_root=$TRAIN_ROOT
elif [ "$1"x = "test"x ]
then
    RUNTIME_NAME=$2
    GPU=$3
	USE_SEG=$4
    echo "Test for $RUNTIME_NAME on $GPU ..."
    python workshop.py --cmd=test --conf_name=$RUNTIME_NAME --gpu=$GPU --use_seg=$USE_SEG --train_root=$TRAIN_ROOT
elif [ "$1"x = "visualize"x ]
then
    RUNTIME_NAME=$2
    VISUAL_DATA=$3
	GPU=$4
    echo "Visualization for $RUNTIME_NAME form data $VISUAL_DATA ..."
    python workshop.py --cmd=visualize --conf_name=$RUNTIME_NAME --gpu=$GPU --visualize_file=$VISUAL_DATA --train_root=$TRAIN_ROOT
fi
