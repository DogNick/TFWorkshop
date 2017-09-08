import os
import vae
import seq2seq 
import score
from Config import confs
from Config import Config

########## External ###########

# Tsinghua
confs["tsinghua"] = Config()
confs["tsinghua"].model_kind                        = "Tsinghua"
confs["tsinghua"].input_max_len                     = 40
confs["tsinghua"].output_max_len                    = 12 
confs["tsinghua"].beam_size                         = 5 
confs["postprob"] = Config() 
confs["postprob"].model_kind                        = "Postprob"
confs["postprob"].input_max_len                     = 40 
confs["postprob"].output_max_len                    = 30 

# tianchuan
confs["tianchuan_cnn"] = Config()
confs["tianchuan_cnn"].model_kind = "KimCNN"

SERVER_SCHEDULES = { 
    "generate":[
        {
            "attn-s2s-all-downsample-addmem":{"tf_server":"0.0.0.0:10010","deploy_gpu":0, "max_in":14, "max_out":14, "max_res":5}
        },
        {
            "vae-1024-attn-addmem":{"tf_server":"0.0.0.0:10011","deploy_gpu":0, "max_in":15, "max_out":15, "max_res":15, "beam_splits":[6,6,6,6,6,6,6,6,6,6]}
        },
        {
            "vae-1024-attn-addmem":{"tf_server":"0.0.0.0:10012","deploy_gpu":1, "max_in":15, "max_out":15, "max_res":10, "beam_splits":[6,6,6,6,6,6,6,6,6,6]}
        },
        {
            "vae-reddit-addmem":{"tf_server":"0.0.0.0:10012","deploy_gpu":1, "max_in":15, "max_out":15, "max_res":10, "beam_splits":[4,4,4,4,4,4,4,4]}
        }
    ],
    "tsinghua":[
        {
            "tsinghua":{"tf_server":"0.0.0.0:10006","deploy_gpu":0, "export":False},
            "postprob":{"tf_server":"0.0.0.0:10005","deploy_gpu":0, "export":False}
        }
    ],

    "matchpoem":[
        {
            "attn-bi-poem-no-ave-len":{"tf_server":"0.0.0.0:10020","deploy_gpu":0, "max_in":10, "max_out":10, "max_res":35, "beam_splits":[8,8,8,8,8,8,8,8,8,8,8,8,8]}
        },
        {
            "attn-bi-poem-no-ave-len":{"tf_server":"0.0.0.0:10021","deploy_gpu":1, "max_in":10, "max_out":10, "max_res":35, "beam_splits":[8,8,8,8,8,8,8,8,8,8,8,8,8]}
        }
    ],
    "judgepoem":[
        {
            "rnncls-bi-judge_poem":{"tf_server":"0.0.0.0:10030","deploy_gpu":0},
        },
        {
            "rnncls-bi-judge_poem":{"tf_server":"0.0.0.0:10031","deploy_gpu":1},
        }
    ],
    "intent":[
        {
            "tianchuan_cnn":{"tf_server":"0.0.0.0:10040","deploy_gpu":1},
        }
    ],
	"chaten":[
        {
            "cvae-noattn-opensubtitle_gt3":{"tf_server":"0.0.0.0:10050","deploy_gpu":7},
            "news2s-opensubtitle_gt3_reverse":{
				"tf_server":"0.0.0.0:10051",
				"deploy_gpu":7,
				"variants":"score",
				"lmda":0.3,
				"alpha":0.6,
				"beta":0.1,
			},
        }
    ],
	"scorer":[
		{"cvae-noattn-opensubtitle_gt3-emb":{"tf_server":"0.0.0.0:10060","deploy_gpu":6}}
	]
}

DESC = { 
    "tsinghua":"tsinghua plan use generate first" \
            "and reverse post prob for reranking", 
    "generate":"Nick plan use attn-addmem, vae," \
            "attn-s2s and segmented beam_search " \
            "to generate and rerank with GNMT score",
    "matchpoem": "for fun",
    "judgepoem": "to pickup all possible poem sentences from one query and show the probs",
	"chaten": "cvae-noattn + posterior probability score"
}

####################################################################
if __name__ == "__main__":
    print ("\n=========  All Configuration ========\n")
    for each in confs:
        print (" %s" % each)
    print ("\n=====================================\n")
