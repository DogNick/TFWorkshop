#!/bin/bash
OM=(\
	root@10.142.100.135 \
	root@10.142.100.135 \
)

DEST_DIR=/search/odin/dialogue/chatbot/trunk/scripts
SCRIPT=qqgroup_start.sh



if [ $# -eq 1 ]
then
	id=`echo $1 | bc 2>/dev/null`
	if [ $id != $1 ]
	then
		echo ""
		echo "ERROR !"
		echo "Must input number from 0 to +inf when only one arg is offered"
		echo " "
		exit
	else
		ssh ${OM[$1]} "cd $DEST_DIR; sh $SCRIPT; sh tailf.sh"
	fi
elif [ $# -eq 2 -a $1"x" = "see"x ]
then
	echo ${OM[$2]}
	ssh ${OM[$2]} "cd $DEST_DIR; sh tailf.sh"
fi
