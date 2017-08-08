from __future__ import absolute_import
from __future__ import print_function 
from __future__ import division

import re
import os
import time
import sys
import math
import grpc
import shutil
import codecs
import traceback
import numpy as np

sys.path.append("models")
#sys.path.insert(0, "/search/odin/Nick/_python_build2")
import logging as log

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.tensorboard.plugins import projector

from shutil import copyfile
from six.moves import xrange  # pylint: disable=redefined-builtin

# Here from my own's models
from models import confs
from models import SERVER_SCHEDULES 
#from models.DynAttnSeq2Seq import *
from models import create 

EOS_ID = 2
graphlg = log.getLogger("graph")
trainlg = log.getLogger("train")

# Training process control
tf.app.flags.DEFINE_string("conf_name", "default", "configuration name")
tf.app.flags.DEFINE_string("train_root", "runtime", "Training root directory.")
tf.app.flags.DEFINE_string("params_path", "", "specify the model parameters to restore from to test a model")
tf.app.flags.DEFINE_string("gpu", None, "specify the gpu to use")

tf.app.flags.DEFINE_integer("steps_per_print", 10, "How many training steps to do per print")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400, "How many training steps to do per checkpoint.")


tf.app.flags.DEFINE_string("job_type", "", "ps or worker")
tf.app.flags.DEFINE_integer("task_id", 0, "task id")
tf.app.flags.DEFINE_boolean("export", False, "to export the conf_name model")
tf.app.flags.DEFINE_string("service", None, "to export service")
tf.app.flags.DEFINE_string("visualize_file", None, "datafile to visualize")
tf.app.flags.DEFINE_string("visualize_name", "Visualize", "name")
tf.app.flags.DEFINE_integer("schedule", None, "to export all models used in schedule")


FLAGS = tf.app.flags.FLAGS

def main(_):
	# Get common params from gflags, get conf from confs
	job_type = FLAGS.job_type
	task_id = FLAGS.task_id
	conf_name = FLAGS.conf_name
	ckpt_dir = os.path.join(FLAGS.train_root, conf_name)
	gpu = FLAGS.gpu

	# Visualization 
	if FLAGS.visualize_file != None:
		visual_name = FLAGS.visualize_name 
		with codecs.open(FLAGS.visualize_file) as f:
			records = [re.split("\t", line.strip())[0] for line in f]
		model = create(conf_name, job_type=job_type, task_id=task_id)
		model.visualize(FLAGS.train_root, gpu=gpu, records=records)

	# Export for deployment
	elif FLAGS.export == True: 
		if FLAGS.schedule != None:
			schedule = SERVER_SCHEDULES[FLAGS.service][FLAGS.schedule]
		else:
			schedule["Null"] = {conf_name:{}}
		for conf_name in schedule:
			if conf_name not in confs:
				print("\nNo model conf '%s' found !!!! Skipped\n" % conf_name)
				exit(0)
			if schedule[conf_name].get("export", True) == False:
				continue

			# adjust config for deploy goal
			conf = confs[conf_name]

			conf.output_max_len = schedule[conf_name].get("max_out", conf.output_max_len)
			conf.input_max_len = schedule[conf_name].get("max_in", conf.input_max_len)
			conf.max_res_num = schedule[conf_name].get("max_res", conf.max_res_num)
			conf.beam_splits = schedule[conf_name].get("beam_splits", conf.beam_splits)
			conf.stddev = schedule[conf_name].get("stddev", conf.stddev)
			conf.output_keep_prob = 1.0

			# do it
			model = create(conf_name, job_type=job_type, task_id=task_id)
			model.export(conf_name, FLAGS.train_root, "servers/deployments")
			tf.reset_default_graph()
	# Train (distributed or single)
	else:
		model = create(conf_name, job_type=job_type, task_id=task_id)
		if model.conf.cluster and job_type == "worker" or job_type == "single":
			spp = FLAGS.steps_per_print
			spc = FLAGS.steps_per_checkpoint

			# Build graph, initialize graph and creat supervisor 
			sess, graph_nodes = model.init_monitored_train(FLAGS.train_root, gpu)
			data_time, step_time, loss = 0.0, 0.0, 0.0
			trainlg.info("Main loop begin..")
			offset = 0 
			iters = 0
			while not model.sess.should_stop():
				# Data preproc 
				start_time = time.time()
				examples = model.fetch_data(use_random=False, begin=offset, size=model.conf.batch_size)
				input_feed = model.preproc(examples, for_deploy=False, use_seg=False, default_wgt=1.5)
				data_time += (time.time() - start_time) / spp
				if iters % spp == 0:
					trainlg.info("Data preprocess time %.5f" % data_time)
					data_time = 0.0
				step_out = sess.run({"loss":graph_nodes["loss"], "update":graph_nodes["update"]}, input_feed)
				offset = (offset + model.conf.batch_size) % len(model.train_set) if not model.conf.use_data_queue else 0
				iters += 1
		elif model.conf.cluster and job_type == "ps":
			model.join_param_server()
		else:
			print ("Really don't know what you want...")

if __name__ == "__main__":
  tf.app.run()
