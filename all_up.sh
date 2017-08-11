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

###################################################################3
#sh job.sh create cvae-1024-simpleprior-attn single 0 1
#sleep 1.0
#sh job.sh see cvae-1024-simpleprior-attn
#sh job.sh create cvae-1024-simpleprior-attn-addmem single 0 2
#sh job.sh create cvae-128-simpleprior-attn single 0 3

#sh job.sh create cvae-512-noprior-noattn single 0 0
#sh job.sh create cvae-512-noprior-attn single 0 3
#sh job.sh create cvae-512-prior-attn single 0 5
#sh job.sh create cvae-512-prior-noattn single 0 6
#sh job.sh create vae-1024-attn-addmem single 0 6 

#sh job.sh create vae-bi-1024-attn-addmem-poem single 0 5
#sh job.sh create attn-bi-poem-no-ave-len single 0 1 
#sh job.sh create attn-s2s-all-downsample-n-gram-addmem single 0 7 

#sh job.sh create rnncls-bi-judge_poem single 0 0 
#sleep 1.0
#sh job.sh see rnncls-bi-judge_poem


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

sh job.sh create vae2-opensubtitle single 0 2
sleep 1.0
sh job.sh see vae2-opensubtitle
#sh job.sh create cvae-simpleprior-reddit-addmem single 0 1
#sleep 1.0
#sh job.sh see cvae-simpleprior-reddit-addmem
