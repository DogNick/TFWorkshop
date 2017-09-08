#!/bin/bash

# A quick startup

##############################################################
#sh job.sh create allattn single 0 0 
#sleep 1.0
#sh job.sh see allattn 

#sh job.sh create allattn_dist ps 0 0 
#sh job.sh create allattn_dist ps 1 0 
#sh job.sh create allattn_dist ps 2 0 
#
#sh job.sh create allattn_dist worker 0 1 
#sh job.sh create allattn_dist worker 1 2 
#sh job.sh create allattn_dist worker 2 3 
#sh job.sh create allattn_dist worker 3 4 
##sh job.sh create allattn_dist worker 4 0 
##sh job.sh create allattn_dist worker 5 0 
##sh job.sh create allattn_dist worker 6 0 
#
#sleep 1.0
#sh job.sh see allattn_dist


#sh job.sh create cvaeattn-subtitle_gt3_joint_prime_clean single 0 0 
#sleep 1.0
#sh job.sh see cvaeattn-subtitle_gt3_joint_prime_clean 

#sh job.sh create vaeattn-opensubtitle_gt3_joint_prime single 0 1 
#sleep 1.0
#sh job.sh see vaeattn-opensubtitle_gt3_joint_prime 

#sh job.sh create news2s-opensubtitle_gt3_reverse single 0 2 
#sleep 1.0
#sh job.sh see news2s-opensubtitle_gt3_reverse 

#sh job.sh create news2s-opensubtitle_gt3_joint_prime single 0 3 
#sleep 1.0
#sh job.sh see news2s-opensubtitle_gt3_joint_prime 

#sh job.sh create vaeattn-opensubtitle_gt3 single 0 4 
#sleep 1.0
#sh job.sh see vaeattn-opensubtitle_gt3 

#sh job.sh create news2s-opensubtitle_gt3_joint_reverse single 0 5 
#sleep 1.0
#sh job.sh see news2s-opensubtitle_gt3_joint_reverse 

#sh job.sh create cvae-noattn-opensubtitle_gt3 single 0 6 
#sleep 1.0
#sh job.sh see cvae-noattn-opensubtitle_gt3 

sh job.sh create cvaeattn2-weibo-bought single 0 0 
sleep 1.0
sh job.sh see cvaeattn2-weibo-bought 


