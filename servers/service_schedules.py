# All servables schedules
SERVICE_SCHEDULES = { 
    "generate":[
        {
			"servables":[
				{"model":"attn-s2s-all-downsample-addmem","tf_server":"0.0.0.0:10010","deploy_gpu":0, "max_in":14, "max_out":14, "max_res":5}
			],
			"nodes":[]
        },
		{
			"servables":[
				{"model":"vae-1024-attn-addmem","tf_server":"0.0.0.0:10011","deploy_gpu":0, "max_in":15, "max_out":15, "max_res":15, "beam_splits":[6,6,6,6,6,6,6,6,6,6]}
			],
			"nodes":[]
		},
		{
			"servables":[
				{"model":"vae-1024-attn-addmem","tf_server":"0.0.0.0:10012","deploy_gpu":1, "max_in":15, "max_out":15, "max_res":10, "beam_splits":[6,6,6,6,6,6,6,6,6,6]}
			],
			"nodes":[]
		},
        {
			"servables":[
				{"model":"vae-reddit-addmem","tf_server":"0.0.0.0:10012","deploy_gpu":1, "max_in":15, "max_out":15, "max_res":10, "beam_splits":[4,4,4,4,4,4,4,4]}
			],
			"nodes":[]
        },
        {
			"servables":[
            	{"model":"cvaeattn2-weibo-bought","tf_server":"0.0.0.0:10013","deploy_gpu":5,"max_in":15,"max_out":20,"max_res":60,"beam_splits":[6,6,6,6,6,6,6,6,6]},
				{"model":"cvaeattn2-512-weibo-bought","tf_server":"0.0.0.0:10014","deploy_gpu":6,"max_in":15,"max_out":20,"max_res":60,"beam_splits":[6,6,6,6,6,6,6,6,6]},
				{"model":"news2s-weibo-bought_reverse","tf_server":"0.0.0.0:10015","deploy_gpu":7,"variants":"score","lmda":0.4,"alpha":0.6,"beta":0.1,}
        	],
			"params":{},
			"nodes":[]
		}
	],
    "matchpoem":[
		{
			"servables":[
				{"model":"attn-bi-poem-no-ave-len","tf_server":"0.0.0.0:10020","deploy_gpu":0, "max_in":10, "max_out":10, "max_res":35, "beam_splits":[8,8,8,8,8,8,8,8,8,8,8,8,8]},
			],
			"nodes":[]
		},
		{
			"servables":[
				{"model":"attn-bi-poem-no-ave-len","tf_server":"0.0.0.0:10021","deploy_gpu":1, "max_in":10, "max_out":10, "max_res":35, "beam_splits":[8,8,8,8,8,8,8,8,8,8,8,8,8]}
			],
			"nodes":[]
		}
    ],
    "judgepoem":[
		{
			"servables":[
				{"model":"rnncls-bi-judge_poem","tf_server":"0.0.0.0:10030","deploy_gpu":0},
			],
			"nodes":[]
        },
		{
			"servables":[
				{"model":"rnncls-bi-judge_poem","tf_server":"0.0.0.0:10031","deploy_gpu":1}
			],
			"nodes":[]
        }
    ],
    "intent":[
        {
			"servables":[
            	{"model":"tianchuan_cnn_tag","tf_server":"0.0.0.0:10041","deploy_gpu":1}
			]	
            #"tianchuan_cnn":{"tf_server":"0.0.0.0:10040","deploy_gpu":1},
        }
    ],
	"chaten":[
        {
			"servables":[	
            	#{"model":"cvae-noattn-opensubtitle_gt3","tf_server":"0.0.0.0:10050","deploy_gpu":7},
				#{"model":"news2s-opensubtitle_gt3_reverse", "tf_server":"0.0.0.0:10051","deploy_gpu":7,"variants":"score","lmda":0.4, "alpha":0.6, "beta":0.1},
				{"model":"cvaeattn-subtitle_gt3_joint_prime_clean", "tf_server":"0.0.0.0:10050", "deploy_gpu":7, "joint_prime":True, "max_res":25},
				{"model":"news2s-opensubtitle_gt3_joint_reverse", "tf_server":"0.0.0.0:10051", "deploy_gpu":6, "variants":"score", "lmda":0.4, "alpha":0.6, "beta":0.1, "joint_prime":True}
			],
			"nodes":[]	
        }
    ],
	"scorer":[
		{
			"servables":[
				{"model":"cvae-noattn-opensubtitle_gt3-emb", "tf_server":"0.0.0.0:10060", "deploy_gpu":6},
				{"model":"news2s-opensubtitle_gt3_joint_reverse", "tf_server":"0.0.0.0:10052", "deploy_gpu":7, "variants":"score", "lmda":0.4, "alpha":0.6, "beta":0.1, "joint_prime":True}
			],
			"nodes":[]
		}
	]
}

# Some descriptions of different servables
DESC = { 
    "tsinghua":"tsinghua plan use generate first" \
            "and reverse post prob for reranking", 
    "generate":"Nick plan use attn-addmem, vae," \
            "attn-s2s and segmented beam_search " \
            "to generate and rerank with GNMT score",
    "matchpoem": "for fun",
    "judgepoem": "to pickup all possible poem sentences from one query and show the probs",
	"chaten": "cvae-noattn + posterior probability score",
	"cvae-generate": "cvaeattn2, z on enc_states and attention,  posterior probability score is comming soon"
}

