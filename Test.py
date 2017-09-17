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
	#name = "test_data/fangfei_lt10_0401_formalized"
	name = "test_data/chat_diff"
	#name = "test_data/fangfei_lt10_0401_jointprime"
	#name = "test_data/fangfei_lt10_0401_formalized_ask"
	#name = "test_data/fangfei_lt10_0401_jointprime_ask"

	# get graph configuration
	runtime_names = [
		#("news2s-opensubtitle_gt3", None)
		#("vaeattn-opensubtitle_gt3", None)
		#("vaeattn-opensubtitle_gt3_joint_prime", None)
		#("news2s-opensubtitle_gt3_joint_prime", None)
		#("news2s-noinit-opensubtitle_gt3", None)
		#("news2s-twitter", None)
		#("news2s-twitter-clean", 89401)
		#("cvae-noattn-opensubtitle_gt3", None)
		("cvaeattn2-weibo-bought", None)
		#("cvaeattn-subtitle_gt3_joint_prime_clean", None)
		#("news2s-opensubtitle_gt3_joint_prime", None)
	]
	scorer_names = [ 
		#("news2s-opensubtitle_gt3_reverse",  None)
		#("news2s-opensubtitle_gt3_joint_reverse",  None)
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
		model.conf.variants = "score"
		sess, graph_nodes, global_steps = init_inference(runtime_root=train_root, model_core=model, gpu=FLAGS.gpu, ckpt_steps=ckpt_steps)
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
			batch_res_of_all_models.append(out_after_proc)

		for k, example_res in enumerate(zip(*batch_res_of_all_models)):
			all_model_res = []
			[ all_model_res.extend(each) for each in example_res]

			r = 0.4
			outputs = [each["outputs"] for each in all_model_res]
			names = [each["model_name"] for each in all_model_res]

			#probs = [each["probs"] for each in all_model_res]
			# Use Model scoring	
			pairs = ["%s\t%s" % (batch[k], " ".join(each)) for each in outputs]
			if len(pairs) == 0:
				continue
			scores = []
			for score_model_name in scorer:
				score_sess, score_graph_nodes, score_model = scorer[score_model_name]
				score_input_feed = score_model.preproc(pairs, use_seg=(FLAGS.lan=="ch"), for_deploy=True)
				score_out = score_sess.run(score_graph_nodes["outputs"], score_input_feed)
				scores.append(score_out["logprobs"])

			posteriors = [sum(each) for each in zip(*scores)]
			new_probs = [each["probs"] for each in all_model_res]  

			scored_res = Nick_plan.score_with_prob_attn(outputs, new_probs, None, posteriors, r, alpha=0.6, beta=0.2, is_ch=(FLAGS.lan=="ch"), average_across_len=False)

			for l, each in enumerate(names):
				final, infos = scored_res[l]
				infos["model_name"] = each

			final_res = sorted(scored_res, key=lambda x:x[1]["score"], reverse=True)

			if FLAGS.command == "see":
				print "==================================="
				print records[i + k]
				for key, v in input_feed.items():
					print key, v[k]

				print "==================================="
				for each in final_res[0:35]:
					final, infos = each
					print final, infos 
				raw_input()
			elif FLAGS.command == "dump":
				if len(final_res) == 0:
					print "[NO results] %s" % records[i + k] 
				else:
					print "%s\t%s" % (records[i + k], final_res[0])
			elif FLAGS.command == "dumpall":
				for final, infos in final_res:
					print "%s\t%s\t%s" % (records[i + k], final, infos)
			else:
				print "command dump or see"

if __name__ == "__main__": 
	main()
