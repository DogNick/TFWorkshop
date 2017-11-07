from __future__ import absolute_import
#from __future__ import print_function 
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

from shutil import copyfile
from six.moves import xrange  # pylint: disable=redefined-builtin

# Here from my own's models
from models import confs
from servers import SERVICE_SCHEDULES 
from models import create, join_param_server, init_inference, init_dummy_train, init_monitored_train  

EOS_ID = 2
graphlg = log.getLogger("graph")
trainlg = log.getLogger("train")

# General params 
tf.app.flags.DEFINE_string("conf_name", "default", "configuration name")
tf.app.flags.DEFINE_string("train_root", "runtime", "Training root directory.")
tf.app.flags.DEFINE_string("gpu", 0, "specify the gpu to use")


# Command line to utilize workshop functions
tf.app.flags.DEFINE_string("cmd", "dummytrain", "for command-line functions"
							 "like [train, dummytrain, test, export, visualize")

# for train job
tf.app.flags.DEFINE_string("job_type", "single", "ps or worker")
tf.app.flags.DEFINE_integer("task_id", 0, "task id")
tf.app.flags.DEFINE_integer("steps_per_print", 10, "How many training steps to do per print")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400, "steps to take to make a checkpoint")
# for test
tf.app.flags.DEFINE_string("variants", "", "model variants when testing")
tf.app.flags.DEFINE_boolean("use_seg", True, "weather to use chinese word segment")
tf.app.flags.DEFINE_string("ckpt_steps", "", "model of skept_steps to restore")

# for export
tf.app.flags.DEFINE_string("service", None, "to export service")
tf.app.flags.DEFINE_integer("schedule", None, "to export all models used in schedule")
# for visualization
tf.app.flags.DEFINE_string("visualize_file", None, "datafile to visualize")
tf.app.flags.DEFINE_integer("max_line", -1, "datafile to visualize")


FLAGS = tf.app.flags.FLAGS

def main(_):
	# Visualization 
	if FLAGS.cmd == "visualize":
		model = create(FLAGS.conf_name, job_type=FLAGS.job_type, task_id=FLAGS.task_id)
		print "Reading data..."
		with codecs.open(FLAGS.visualize_file) as f:
			count = 0
			records = []
			for line in f:
				count += 1
				line = line.strip()
				records.append(re.split("\t", line)[0])
				if FLAGS.max_line > 0 and count >= FLAGS.max_line: 
					break
		sess, graph_nodes, ckpt_steps = init_inference(runtime_root=FLAGS.train_root, model_core=model, gpu=FLAGS.gpu)
		model.visualize(FLAGS.train_root, sess, graph_nodes, records, FLAGS.use_seg)
	# Export for deployment
	elif FLAGS.cmd == "export": 
		if FLAGS.schedule != None:
			schedule = SERVICE_SCHEDULES[FLAGS.service][FLAGS.schedule]
		else:
			schedule["servables"] = [] 
		for model_conf in schedule["servables"]:
			conf_name = model_conf["model"]
			model = create(conf_name, job_type=FLAGS.job_type, task_id=FLAGS.task_id)
			if conf_name not in confs:
				print("\nNo model conf '%s' found !!!! Skipped\n" % conf_name)
				exit(0)

			if model_conf.get("export", True) == False:
				continue

			model = create(conf_name, job_type="single", task_id=0)
			model.apply_deploy_conf(model_conf)	

			ckpt_steps = model_conf.get("ckpt_steps", None) 
			gpu = model_conf.get("deploy_gpu", 0)
			sess, graph_nodes, ckpt_steps = init_inference(runtime_root=FLAGS.train_root, model_core=model, gpu=gpu, ckpt_steps=ckpt_steps)

			# do it
			model.export(sess=sess, nodes=graph_nodes, version=ckpt_steps, deploy_dir="servers/deployments")
			tf.reset_default_graph()

	# Train (distributed or single)
	elif FLAGS.cmd == "dummytrain": 
		model = create(FLAGS.conf_name, job_type=FLAGS.job_type, task_id=FLAGS.task_id)
		sess, graph_nodes = init_dummy_train(runtime_root=FLAGS.train_root, model_core=model, gpu=FLAGS.gpu)
		model.dummy_train(sess, graph_nodes)

	elif FLAGS.cmd == "test":
		model = create(FLAGS.conf_name, job_type=FLAGS.job_type, task_id=FLAGS.task_id)
		model.conf.variants = FLAGS.variants

		sess, graph_nodes, ckpt_steps = init_inference(runtime_root=FLAGS.train_root, model_core=model, gpu=FLAGS.gpu, ckpt_steps=FLAGS.ckpt_steps)
		model.test(sess, graph_nodes, use_seg=FLAGS.use_seg)

	elif FLAGS.cmd == "train":
		model = create(FLAGS.conf_name, job_type=FLAGS.job_type, task_id=FLAGS.task_id)
		if model.conf.cluster and FLAGS.job_type == "worker" or FLAGS.job_type == "single":
			# Build graph, initialize graph and creat supervisor 
			sess, graph_nodes = init_monitored_train(runtime_root=FLAGS.train_root, model_core=model, gpu=FLAGS.gpu)
			data_time, step_time, loss = 0.0, 0.0, 0.0
			trainlg.info("Main loop begin..")
			offset = 0 
			iters = 0
			while not sess.should_stop():
				# Data preproc 
				start_time = time.time()
				examples = model.fetch_data(use_random=True, begin=offset, size=model.conf.batch_size)
				input_feed = model.preproc(examples, for_deploy=False, use_seg=False, default_wgt=1.5)
				data_time += (time.time() - start_time) / FLAGS.steps_per_print
				if iters % FLAGS.steps_per_print == 0:
					trainlg.info("Data preprocess time %.5f" % data_time)
					data_time = 0.0
				step_out = sess.run({"loss":graph_nodes["loss"], "update":graph_nodes["update"]}, input_feed)
				offset = (offset + model.conf.batch_size) % len(model.train_set) if not model.conf.use_data_queue else 0
				iters += 1
		elif model.conf.cluster and FLAGS.job_type == "ps":
			join_param_server(model.conf.cluster, model.job_type, model.task_id)
		else:
			print ("Some errors in cluster, job_type or task_id configuration")
	else:
		print ("Really don't know what you want...")


if __name__ == "__main__":
  tf.app.run()
