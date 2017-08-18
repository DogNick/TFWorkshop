#!/bin/bash

# A quick startup

#sh job.sh create cvae2-merge-stc-weibo single 0 0 
#sleep 1.0
#sh job.sh see cvae2-merge-stc-weibo

#sh job.sh create cvae2_1-merge-stc-weibo single 0 1 
#sleep 1.0
#sh job.sh see cvae2_1-merge-stc-weibo

#sh job.sh create cvae3-merge-stc-weibo single 0 7   
#sleep 1.0
#sh job.sh see cvae3-merge-stc-weibo

#sh job.sh create cvae4-merge-stc-weibo single 0 5 
#sleep 1.0
#sh job.sh see cvae4-merge-stc-weibo

#sh job.sh create cvae4_1-weibo-stc-bought single 0 4 
#sleep 1.0
#sh job.sh see cvae4_1-weibo-stc-bought

#sh job.sh create attn-bi-s2s-all-downsample-addmem single 0 3 
#sleep 1.0
#sh job.sh see attn-bi-s2s-all-downsample-addmem 

#sh job.sh create attn-bi-s2s-all-downsample-addmem2 single 0 2 
#sleep 1.0
#sh job.sh see attn-bi-s2s-all-downsample-addmem2 

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

#sh job.sh create vae-reddit-addmem single 0 0 
#sleep 1.0
#sh job.sh see vae-reddit-addmem 


#sh job.sh create news2s-opensubtitle_gt3 single 0 6 
#sleep 1.0
#sh job.sh see news2s-opensubtitle_gt3 

#sh job.sh create news2s-twitter single 0 7 
#sleep 1.0
#sh job.sh see news2s-twitter

sh job.sh create vae2-1024-attn-addmem single 0 0 
sleep 1.0
sh job.sh see vae2-1024-attn-addmem 

#sh job.sh create cvae-opensubtitle single 0 1 
#sleep 1.0
#sh job.sh see cvae-opensubtitle


#sh job.sh create cvae-no-attn-opensubtitle_gt3 single 0 3 
#sleep 1.0
#sh job.sh see cvae-no-attn-opensubtitle_gt3 


#sh job.sh create cvae-noattn-opensubtitle_gt3 single 0 3 
#sleep 1.0
#sh job.sh see cvae-noattn-opensubtitle_gt3 

#sh job.sh create attns2s-opensubtitle_gt3 single 0 4 
#sleep 1.0
#sh job.sh see attns2s-opensubtitle_gt3 

#sh job.sh create attns2s-twitter single 0 5
#sleep 1.0
#sh job.sh see attns2s-twitter 
