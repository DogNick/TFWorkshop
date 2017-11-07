import os
import re
import select
import sys
import gflags
import subprocess
import signal

gflags.DEFINE_string("schedule_name", None, "service:schedule")
gflags.MarkFlagAsRequired('schedule_name') 

FLAGS = gflags.FLAGS

ONLINE_ALL = [
	"root@10.153.50.79:/search/odin/offline/Workshop", 
	"root@10.153.50.80:/search/odin/offline/Workshop",
	"root@10.153.50.81:/search/odin/offline/Workshop",
	"root@10.153.50.82:/search/odin/offline/Workshop",
	"root@10.153.50.83:/search/odin/offline/Workshop",
	"root@10.153.50.84:/search/odin/offline/Workshop",
	"root@10.153.58.66:/search/odin/offline/Workshop",
	"root@10.153.58.67:/search/odin/offline/Workshop",
	#"root", "10.141.105.100", "/search/odin/offline/Workshop"),
	#"root", "10.141.105.106", "/search/odin/offline/Workshop")
]
SERVICE_SCHEDULES = { 
    "generate":{
        "old":{
			"servables":[
				{"model":"attn-s2s-all-downsample-addmem","tf_server":"0.0.0.0:10010","deploy_gpu":0, "max_in":14, "max_out":14, "max_res":5}
			],
			"nodes":[],
			"desc":"A very old single generation model attn-s2s-all-downsample-addmem"
        },
		"old_vae_0":{
			"servables":[
				{"model":"vae-1024-attn-addmem","tf_server":"0.0.0.0:10011","deploy_gpu":0, "max_in":15, "max_out":15, "max_res":15, "beam_splits":[6,6,6,6,6,6,6,6,6,6]}
			],
			"nodes":[],
			"desc":"An old chinese generation model with VAE framework and attention mechanism powered by addmem, on GPU 0"
		},
		"old_vae_1":{
			"servables":[
				{"model":"vae-1024-attn-addmem","tf_server":"0.0.0.0:10012","deploy_gpu":1, "max_in":15, "max_out":15, "max_res":10, "beam_splits":[6,6,6,6,6,6,6,6,6,6]}
			],
			"nodes":[],
			"desc":"Another old chinese generate model with VAE framework and attention mechanism powered by addmem, on GPU 1"
		},
        "old_vae_en":{
			"servables":[
				{"model":"vae-reddit-addmem","tf_server":"0.0.0.0:10012","deploy_gpu":1, "max_in":15, "max_out":15, "max_res":10, "beam_splits":[4,4,4,4,4,4,4,4]}
			],
			"nodes":[],
			"desc":"An english generator with a single model of vae framework"
        },
        "cvae_posterior_cn":{
			"servables":[
            	{"model":"cvaeattn2-weibo-bought","tf_server":"0.0.0.0:10013","deploy_gpu":5,"max_in":15,"max_out":20,"max_res":60,"beam_splits":[6,6,6,6,6,6,6,6,6]},
				{"model":"cvaeattn2-512-weibo-bought","tf_server":"0.0.0.0:10014","deploy_gpu":6,"max_in":15,"max_out":20,"max_res":60,"beam_splits":[6,6,6,6,6,6,6,6,6]},
				{"model":"news2s-weibo-bought_reverse","tf_server":"0.0.0.0:10015","deploy_gpu":7,"variants":"score","lmda":0.4,"alpha":0.6,"beta":0.1,}
        	],
			"params":{},
			"nodes":ONLINE_ALL,
			"desc":"A composed Chinese generator with two generate models(in parallel) and one posterior scorer"
		},
		"cvae_posterior_en":{
			"servables":[	
            	#{"model":"cvae-noattn-opensubtitle_gt3","tf_server":"0.0.0.0:10050","deploy_gpu":7},
				#{"model":"news2s-opensubtitle_gt3_reverse", "tf_server":"0.0.0.0:10051","deploy_gpu":7,"variants":"score","lmda":0.4, "alpha":0.6, "beta":0.1},
				{"model":"cvaeattn-subtitle_gt3_joint_prime_clean", "tf_server":"0.0.0.0:10050", "deploy_gpu":7, "joint_prime":True, "max_res":25},
				{"model":"news2s-opensubtitle_gt3_joint_reverse", "tf_server":"0.0.0.0:10051", "deploy_gpu":6, "variants":"score", "lmda":0.4, "alpha":0.6, "beta":0.1, "joint_prime":True}
			],
			"nodes":[],
			"desc":"A composed English generator with one generate model and one posterior scorer"
        }
	},
    "matchpoem":{
		"bi_attn_0":{
			"servables":[
				{"model":"attn-bi-poem-no-ave-len","tf_server":"0.0.0.0:10020","deploy_gpu":0, "max_in":10, "max_out":10, "max_res":35, "beam_splits":[8,8,8,8,8,8,8,8,8,8,8,8,8]},
			],
			"nodes":[],
			"desc":"An old poem generator with bi-lstm"
		},
		"bi_attn_1":{
			"servables":[
				{"model":"attn-bi-poem-no-ave-len","tf_server":"0.0.0.0:10021","deploy_gpu":1, "max_in":10, "max_out":10, "max_res":35, "beam_splits":[8,8,8,8,8,8,8,8,8,8,8,8,8]}
			],
			"nodes":[],
			"desc":"Another old poem generator with bi-lstm"
		}
    },
    "judgepoem":{
		"bi_rnn_0":{
			"servables":[
				{"model":"rnncls-bi-judge_poem","tf_server":"0.0.0.0:10030","deploy_gpu":0},
			],
			"nodes":[],
			"desc":""
        },
		"bi_rnn_1":{
			"servables":[
				{"model":"rnncls-bi-judge_poem","tf_server":"0.0.0.0:10031","deploy_gpu":1}
			],
			"nodes":[],
			"desc":""
        }
    },
    "intent":{
        "cnn":{
			"servables":[
            	{"model":"tianchuan_cnn_tag","tf_server":"0.0.0.0:10041","deploy_gpu":1}
			],
			"nodes":[],
			"desc":"An intention classifier with cnn done by tianchuan"
            #"tianchuan_cnn":{"tf_server":"0.0.0.0:10040","deploy_gpu":1},
        }
    },
	"scorer":{
		"cvae_emb":{
			"servables":[
				{"model":"cvae-noattn-opensubtitle_gt3-emb", "tf_server":"0.0.0.0:10060", "deploy_gpu":6},
				{"model":"news2s-opensubtitle_gt3_joint_reverse", "tf_server":"0.0.0.0:10052", "deploy_gpu":7, "variants":"score", "lmda":0.4, "alpha":0.6, "beta":0.1, "joint_prime":True}
			],
			"nodes":[],
			"desc":""
		}
	}	
}
if __name__ == "__main__":
	deployments_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deployments")

	FLAGS(sys.argv)
	service_name, schedule_name = re.split(":", FLAGS.schedule_name)
	schedule = SERVICE_SCHEDULES[service_name][schedule_name]
		
	children = []	
	for node_with_workshop_path in schedule["nodes"]:
		for model_conf in schedule["servables"]:
			model_path = os.path.join(deployments_path, model_conf["model"])
			dest_path = os.path.join(node_with_workshop_path, "servers/deployments/")
			proc = subprocess.Popen(['scp', '-r', model_path, dest_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setpgrp)
			children.append(proc)

	def signal_handler(signum, frame):
		for child in children:
			os.killpg(os.getpgid(child.pid), signum)
			print('You killed  child %d' % child.pid)
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)

	try:
		streams = [p.stdout for p in children]

		def output(s):
			sys.stdout.write(s)
			sys.stdout.flush()
		while True:
			rstreams, _, _ = select.select(streams, [], [])
			for stream in rstreams:
				line = stream.readline()
				output(line)
			if all(p.poll() is not None for p in processes):
				break
		for stream in streams:
			output(stream.read())
	except Exception, e:
		print e 
		for child in children:
			os.killpg(os.getpgid(child.pid), signal.SIGTERM)
			print 'You killed  child %d' % child.pid
