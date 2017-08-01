import sys
#sys.path.insert(0, "/search/odin/Nick/_python_build2")
import time
import random
import numpy as np
import tensorflow as tf
from ModelCore import *
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.contrib import lookup
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

from tensorflow.python.training.sync_replicas_optimizer import SyncReplicasOptimizer
from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique
from tensorflow.contrib.rnn import  MultiRNNCell, AttentionCellWrapper, GRUCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.seq2seq.python.ops import loss

from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.util import nest

import Nick_plan
import logging as log

graphlg = log.getLogger("graph")
DynamicAttentionWrapper = dynamic_attention_wrapper.DynamicAttentionWrapper
DynamicAttentionWrapperState = dynamic_attention_wrapper.DynamicAttentionWrapperState 
Bahdanau = dynamic_attention_wrapper.BahdanauAttention
Luong = dynamic_attention_wrapper.LuongAttention

def relu(x, alpha=0.2, max_value=None):
	'''ReLU.
		alpha: slope of negative section.
	'''
	negative_part = tf.nn.relu(-x)
	x = tf.nn.relu(x)
	if max_value is not None:
		x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32), tf.cast(max_value, dtype=tf.float32))
	x -= tf.constant(alpha, dtype=tf.float32) * negative_part
	return x

def FeatureMatrix(conv_conf, inps, scope=None, dtype=tf.float32):
	with variable_scope.variable_scope(scope) as scope: 
		for i, each in enumerate(conv_conf):
			h, w, ci, co = each[0]  
			h_s, w_s = each[1]
			ph, pw = each[2]
			ph_s, pw_s = each[3]
			k = tf.get_variable("filter_%d" % i, [h, w, ci, co], initializer=tf.random_uniform_initializer(-0.4, 0.4))
			conved = tf.nn.conv2d(inps, k, [1, h_s, w_s, 1], padding="SAME")
			#conved = relu(conved)
			conved = tf.nn.tanh(conved)
			# TODO Max pooling (May be Dynamic-k-max-pooling TODO)
			max_pooled = tf.nn.max_pool(value=conved, ksize=[1, ph, pw, 1], strides=[1, ph, pw, 1], data_format="NHWC", padding="SAME") 
			inps = max_pooled
	return inps 

def FC(inputs, h_size, o_size, act):
	fc1 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=h_size, activation_fn=relu,
											weights_initializer=tf.random_uniform_initializer(-0.4, 0.4),
											biases_initializer=tf.random_uniform_initializer(-0.4, 0.4))
	fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=o_size, activation_fn=act,
											weights_initializer=tf.random_uniform_initializer(-0.3, 0.3),
											biases_initializer=tf.random_uniform_initializer(-0.4, 0.4))

	return fc2

def CreateMultiRNNCell(cell_name, num_units, num_layers=1, output_keep_prob=1.0, reuse=False):
	cells = []
	for i in range(num_layers):
		if cell_name == "GRUCell":
			single_cell = GRUCell(num_units=num_units, reuse=reuse)
		elif cell_name == "LSTMCell":
			single_cell = LSTMCell(num_units=num_units, reuse=reuse)
		else:
			graphlg.info("Unknown Cell type !")
			exit(0)
		if output_keep_prob < 1.0:
			single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=output_keep_prob) 
			graphlg.info("Layer %d, Dropout used: output_keep_prob %f" % (i, output_keep_prob))

		#single_cell = DeviceWrapper(ResidualWrapper(single_cell), device='/gpu:%d' % i)
		#single_cell = DeviceWrapper(single_cell, device='/gpu:%d' % i)

		cells.append(single_cell)
	return MultiRNNCell(cells) 


class RNNClassification(ModelCore):
	def __init__(self, name, job_type="single", task_id=0, dtype=tf.float32):
		super(RNNClassification, self).__init__(name, job_type, task_id, dtype) 
		self.embedding = None
		self.out_proj = None
	def build(self, for_deploy, variants=""):
		conf = self.conf
		name = self.name
		job_type = self.job_type
		dtype = self.dtype
		self.beam_size = 1 if (not for_deploy or variants=="score") else sum(self.conf.beam_splits)

		# Input maps
		self.in_table = lookup.MutableHashTable(key_dtype=tf.string,
													 value_dtype=tf.int64,
													 default_value=UNK_ID,
													 shared_name="in_table",
													 name="in_table",
													 checkpoint=True)

		self.enc_str_inps = tf.placeholder(tf.string, shape=(None, conf.input_max_len), name="enc_inps")
		self.enc_lens = tf.placeholder(tf.int32, shape=[None], name="enc_lens")
		self.tags = tf.placeholder(tf.int32, shape=[None, conf.tag_num], name="tags")
		self.down_wgts = tf.placeholder(tf.float32, shape=[None], name="down_wgts")

		# lookup
		self.enc_inps = self.in_table.lookup(self.enc_str_inps)
		#self.enc_inps = tf.Print(self.enc_inps, [self.enc_inps], message="enc_inps", summarize=100000)

		with variable_scope.variable_scope(self.model_kind, dtype=dtype) as scope: 
			# Create encode graph and get attn states
			graphlg.info("Creating embeddings and embedding enc_inps.")
			with ops.device("/cpu:0"):
				self.embedding = variable_scope.get_variable("embedding", [conf.output_vocab_size, conf.embedding_size], initializer=tf.random_uniform_initializer(-0.08, 0.08))
				self.emb_enc_inps = embedding_lookup_unique(self.embedding, self.enc_inps)

			graphlg.info("Creating dynamic rnn...")
			if conf.bidirectional:
				with variable_scope.variable_scope("encoder", dtype=dtype) as scope: 
					cell_fw = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
					cell_bw = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
				self.enc_outs, self.enc_states = bidirectional_dynamic_rnn(
																cell_fw=cell_fw, cell_bw=cell_bw,
																inputs=self.emb_enc_inps,
																sequence_length=self.enc_lens,
																dtype=dtype,
																parallel_iterations=16,
																scope=scope)

				fw_s, bw_s = self.enc_states 
				self.enc_states = []
				for f, b in zip(fw_s, bw_s):
					if isinstance(f, LSTMStateTuple):
						self.enc_states.append(LSTMStateTuple(tf.concat([f.c, b.c], axis=1), tf.concat([f.h, b.h], axis=1)))
					else:
						self.enc_states.append(tf.concat([f, b], 1))
				self.enc_outs = tf.concat([self.enc_outs[0], self.enc_outs[1]], axis=2)
				mem_size = 2 * conf.num_units
				enc_state_size = 2 * conf.num_units 
			else:
				with variable_scope.variable_scope("encoder", dtype=dtype) as scope: 
					cell = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
				self.enc_outs, self.enc_states = dynamic_rnn(cell=cell,
															inputs=self.emb_enc_inps,
															sequence_length=self.enc_lens,
															parallel_iterations=16,
															scope=scope,
															dtype=dtype)
				mem_size = conf.num_units
				enc_state_size = conf.num_units

		self.enc_outs = tf.expand_dims(self.enc_outs, -1)
		with variable_scope.variable_scope("cnn", dtype=dtype, reuse=None) as scope: 
			feature_map = FeatureMatrix(conf.conv_conf, self.enc_outs, scope=scope, dtype=dtype)

		vec = tf.contrib.layers.flatten(feature_map)

		with variable_scope.variable_scope("fc", dtype=dtype, reuse=False) as scope: 
			fc_out = FC(inputs=vec, h_size=conf.fc_h_size, o_size=conf.tag_num, act=relu)
		self.outputs = fc_out

		if not for_deploy:
			#self.tags = tf.Print(self.tags, [self.tags], message="tags", summarize=10000)
			loss = tf.losses.softmax_cross_entropy(self.tags, self.outputs)
			see_loss = loss
			tf.summary.scalar("loss", see_loss)
			self.summary_ops = tf.summary.merge_all()
			self.update = self.backprop(loss)  

			self.train_outputs_map["loss"] = see_loss
			self.train_outputs_map["update"] = self.update

			self.fo_outputs_map["loss"] = see_loss

			self.debug_outputs_map["loss"] = see_loss
			self.debug_outputs_map["outputs"] = self.outputs,
			self.debug_outputs_map["update"] = self.update
			#saver
			self.trainable_params.extend(tf.trainable_variables())
			self.saver = tf.train.Saver(max_to_keep=conf.max_to_keep)
		else:
			if variants == "":
				self.infer_outputs_map["tags"] = tf.nn.softmax(self.outputs)
			else:
				pass

			#saver
			self.trainable_params.extend(tf.trainable_variables())
			self.saver = tf.train.Saver(max_to_keep=conf.max_to_keep)

			# Exporter for serving
			self.model_exporter = exporter.Exporter(self.saver)
			inputs = {
				"enc_inps:0":self.enc_str_inps,
				"enc_lens:0":self.enc_lens
			} 
			outputs = self.infer_outputs_map
			self.model_exporter.init(
				tf.get_default_graph().as_graph_def(),
				named_graph_signatures={
					"inputs": exporter.generic_signature(inputs),
					"outputs": exporter.generic_signature(outputs)
				})
			graphlg.info("Graph done")
			graphlg.info("")
		return

	def get_restorer(self):
		restorer = tf.train.Saver(self.global_params +
								  self.trainable_params + self.optimizer_params +
								  tf.get_default_graph().get_collection("saveable_objects"))
		return restorer

	def get_init_ops(self):
		init_ops = []
		if self.conf.embedding_init:
			init_ops = [tf.variables_initializer(set(self.optimizer_params + self.global_params + self.trainable_params)- set([self.embedding]))]
			w2v = np.load(self.conf.embedding_init)
			init_ops.append(self.embedding.assign(w2v))
		else:
			init_ops = [tf.variables_initializer(set(self.optimizer_params + self.global_params + self.trainable_params))]

		if self.task_id == 0:
			vocab_file = filter(lambda x: re.match("vocab\d+\.all", x) != None, os.listdir(self.conf.data_dir))[0]
			f = codecs.open(os.path.join(self.conf.data_dir, vocab_file))
			k = [line.strip() for line in f]
			k = k[0:self.conf.input_vocab_size]
			v = [i for i in range(len(k))]
			op_in = self.in_table.insert(constant_op.constant(k), constant_op.constant(v, dtype=tf.int64))
			init_ops.extend([op_in])
		return init_ops

	def preproc(self, records, use_seg=True, for_deploy=False, default_wgt=1.0):
		# parsing
		data = []
		for each in records:
			if for_deploy:
				p = each.strip()
				words, _ = tokenize_word(p) if use_seg else (p.split(), None)
				p_list = words #re.split(" +", p.strip())
				data.append([p_list, len(p_list) + 1, [], 1.0])
			else:
				segs = re.split("\t", each.strip())
				if len(segs) < 2:
					continue
				p, tag = segs[0], segs[1]
				p_list = re.split(" +", p)
				tag_list = re.split(" +", tag)

				down_wgts = segs[-1] if len(segs) > 2 else default_wgt 
				data.append([p_list, len(p_list) + 1, tag_list, down_wgts])

		# batching
		conf = self.conf
		batch_enc_inps, batch_enc_lens, batch_tags, batch_down_wgts = [], [], [], []
		for encs, enc_len, tag, down_wgts in data:
			# Encoder inputs are padded, reversed and then padded to max.
			enc_len = enc_len if enc_len < conf.input_max_len else conf.input_max_len
			encs = encs[0:conf.input_max_len]
			if conf.enc_reverse:
				encs = list(reversed(encs + ["_PAD"] * (enc_len - len(encs))))
			enc_inps = encs + ["_PAD"] * (conf.input_max_len - len(encs))

			batch_enc_inps.append(enc_inps)
			batch_enc_lens.append(np.int32(enc_len))
			if not for_deploy:
				# Merge dec inps and targets 
				batch_tags.append(tag)	
				batch_down_wgts.append(down_wgts)
		feed_dict = {
			"enc_inps:0": batch_enc_inps,
			"enc_lens:0": batch_enc_lens,
			"tags:0": batch_tags,
			"down_wgts:0": batch_down_wgts
		}
		for k, v in feed_dict.items():
			if not v: 
				del feed_dict[k]
		return feed_dict

	def after_proc(self, out):
		return out["tags"][0]
		
if __name__ == "__main__":
	name = "rnncls-bi-judge_poem"
	model = RNNClassification(name)
	if len(sys.argv) == 2:
		gpu = 0
	flag = sys.argv[1]
	#model(flag, use_seg=False)
	model(flag, use_seg=False, gpu=1)
