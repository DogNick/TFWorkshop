#coding=utf-8
import sys
sys.path.insert(0, "models")
sys.path.insert(0, "/search/odin/Nick/_python_build2")
from models import create, init_inference 
from models import Nick_plan 
import numpy as np
import tensorflow as tf
import codecs
import copy
import logging as log
import math
from scipy import stats
import re
import os
tf.app.flags.DEFINE_string("gpu", 0, "specify the gpu to use")
tf.app.flags.DEFINE_string("command", "see", "use different test behaviors")
tf.app.flags.DEFINE_string("lan", "ch", "to specify language")
tf.app.flags.DEFINE_integer("batch_size", 100, "to specify language")
FLAGS = tf.app.flags.FLAGS

fh = log.StreamHandler()
fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
log.getLogger("graph").addHandler(fh)
log.getLogger("graph").setLevel(log.DEBUG) 
# create training logger
fh = log.StreamHandler()
fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
log.getLogger("train").addHandler(fh)
log.getLogger("train").setLevel(log.DEBUG)
graphlg = log.getLogger("graph")
trainlg = log.getLogger("train")
train_root = "/search/odin/Nick/GenerateWorkshop/runtime"
def main():
	#name = "data/twitter_25w_filter_len_unk/test"
	#name = "data/opensubtitle_gt3/test"
	name = "test_data/fangfei_lt10_0401_formalized"


	# get graph configuration
	runtime_names = [
		#("news2s-opensubtitle_gt3", None)
		("news2s-noinit-opensubtitle_gt3", None)
		#("news2s-twitter", None)
		#("news2s-twitter-clean", 89401)
		#("cvae-noattn-opensubtitle_gt3", None)
	]
	scorer_names = [ 
		#"attn-bi-s2s-all-downsample-addmem2",
		#"attn-s2s-all-downsample-n-gram-addmem"
		("news2s-opensubtitle_gt3_reverse",  None)
	]

	records = []
	orig_records = []
	with codecs.open(name, "r") as f:
		for line in f:
			line = line.strip()
			orig_records.append(line)
			#line = re.sub(" +", "，", line)
			line = re.split("\t", line)[0]
			records.append(line)

	runtimes = {}
	scorer = {} 
	for model_name, ckpt_steps in runtime_names:
		ckpt_dir = os.path.join(train_root, model_name) 
		model = create(model_name, job_type="single", task_id=0)
		sess, graph_nodes, ckpt_steps = init_inference(runtime_root=train_root, model_core=model, gpu=FLAGS.gpu, ckpt_steps=ckpt_steps)
		runtimes[model_name] = (sess, graph_nodes, model)
		tf.reset_default_graph()

	#for score
	for model_name, ckpt_steps in scorer_names:
		ckpt_dir = os.path.join(train_root, model_name) 
		model = create(model_name, job_type="single", task_id=0)
		model.conf.for_score = True
		sess, graph_nodes, global_steps = model.init_inference(gpu="2", runtime_root=train_root, ckpt_steps)
		scorer[model_name] = (sess, graph_nodes, model)
		tf.reset_default_graph()

	for i in range(0, len(records), FLAGS.batch_size):
		batch_res_of_all_models = []
		batch = records[i:i+FLAGS.batch_size]
		for name, _ in runtime_names:
			sess, graph_nodes, model = runtimes[name]
			input_feed = model.preproc(batch, use_seg=(FLAGS.lan=="ch"), for_deploy=True) 
			step_out = sess.run(graph_nodes["outputs"], input_feed)
			out_after_proc = model.after_proc(step_out)

			# Model scoring
			#for k in range(len(out)):
			#	examples = ["%s\t%s" % (batch[k], " ".join(each)) for each in out[k]]
			#	for score_model_name in scorer:
			#		if score_model_name == name:
			#			continue
			#		input_feed = scorer[score_model_name].preproc(examples, use_seg=is_ch, for_deploy=False)
			#		step_out = scorer[score_model_name].step(input_feed=input_feed, forward_only=True, debug=False, for_deploy=True)
			#		for n, prob in enumerate(step_out["logprobs"]):
			#			probs[k][n] += prob

			# Nick's re-score and some interference
			batch_res = [] 
			for example_res in out_after_proc:
				outputs = [each["outputs"] for each in example_res]
				probs = [each["probs"] for each in example_res]
				scored_res = Nick_plan.score_with_prob_attn(name, outputs, probs, None, alpha=0.8, beta=0.2, is_ch=(FLAGS.lan=="ch"), average_across_len=True)
				batch_res.append(scored_res)
				#batch_res.append([(name, "".join(each[0]), "None", 0, 0, 0, 0, 0, "None")]) 

		batch_res_of_all_models.append(batch_res)

		# Handle each batch of all model res, may merge their results of one example
		for n in range(len(batch_res_of_all_models[0])):
			one_res_of_all_model = []
			for m in range(len(runtime_names)):
				#sorted_res = Nick_plan.rank(batch_res_of_all_models[m][n])
				sorted_res = batch_res_of_all_models[m][n]
				#print sorted_res[0][3].keys()
				#for each in sorted_res:
				#	print each[0], each[1], each[2], "[All:%.5f]" % each[3], "[Org:%.5f]" % each[4], "[LenRatio:%.5f]" % each[5], "[AttnEnt:%.5f]" % each[6]
				#	#print each[0], each[1], each[2], each[3].values()
				one_res_of_all_model.extend(sorted_res)
				#raw_input()
			final_res = sorted(one_res_of_all_model, key=lambda x:x[3], reverse=True)
			#final_res = sorted(one_res_of_all_model, key=lambda x:sum(x[3].values()) /float(len(x[1].decode("utf-8"))), reverse=True)
			if FLAGS.command == "see":
				print "==================================="
				print records[n + i]
				print "==================================="
				for each in final_res[0:80]:
					#print each[0], each[-1], each[2], "[All:%.5f]" % each[3], "[Org:%.5f]" % each[4], "[LenRatio:%.5f]" % each[5], "[AttnEnt:%.5f]" % each[6]
					print each[0], each[1], each[2], each[3]
				raw_input()
			elif FLAGS.command == "dump":
				if len(final_res) == 0:
					print "[NO results] %s" % records[n + i] 
				else:
					print "%s\t%s\t%s" % (records[n + i], final_res[0][1], final_res[0][0])
			elif FLAGS.command == "dumpall":
				for w in range(len(final_res)):
					print "%s\t%s\t%s" % (orig_records[n + i], final_res[w][1], final_res[w][0])
				#raw_input()
			else:
				print "command dump or see"

if __name__ == "__main__": 
	main()
