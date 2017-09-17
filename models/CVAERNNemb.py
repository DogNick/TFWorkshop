#from __future__ import absolute_import
import sys
import time
import math
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
from tensorflow.contrib.rnn import AttentionCellWrapper, GRUCell, LSTMCell, LSTMStateTuple, MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.seq2seq.python.ops import loss

from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.session_bundle import exporter
from components import DecStateInit, AttnCell, DynRNN, CreateMultiRNNCell, CreateVAE 

import Nick_plan
import logging as log

graphlg = log.getLogger("graph")
DynamicAttentionWrapper = dynamic_attention_wrapper.DynamicAttentionWrapper
DynamicAttentionWrapperState = dynamic_attention_wrapper.DynamicAttentionWrapperState 
Bahdanau = dynamic_attention_wrapper.BahdanauAttention
Luong = dynamic_attention_wrapper.LuongAttention

def DecStateInit(raw_states, decoder_cell, batch_size):
	# Encoder states for initial state, with vae 
	with tf.name_scope("DecStateInit"):
		all_states = []
		for each in raw_states:
			if isinstance(each, LSTMStateTuple):
				each = tf.concat([each.c, each.h], 1)
			all_states.append(each)

		concat_all_state = tf.concat(all_states, 1)

		zero_states = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

		if isinstance(zero_states, DynamicAttentionWrapperState):
			cell_states = zero_states.cell_state
		else:
			cell_states = zero_states

		init_states = []
		for i, each in enumerate(cell_states):
			if i > 0:
				init_states.append(each)
			else:
				init_h = tf.layers.dense(concat_all_state, each.h.get_shape()[1], name="ToDecShape")

				if isinstance(each, LSTMStateTuple):
					state = LSTMStateTuple(each.c, init_h)
				else:
					state = init_h 
				init_states.append(state)
	
		if isinstance(decoder_cell,DynamicAttentionWrapper):
			zero_states = DynamicAttentionWrapperState(tuple(init_states), zero_states.attention, zero_states.newmem, zero_states.alignments)
		else:
			zero_states = tuple(init_states)
		
		return zero_states

class CVAERNNemb(ModelCore):
	def __init__(self, name, job_type="single", task_id=0, dtype=tf.float32):
		super(self.__class__, self).__init__(name, job_type, task_id, dtype) 
		self.embedding = None
		self.out_proj = None

	def build_inputs(self, for_deploy):
		inputs = {}
		graphlg.info("Creating placeholders...")
		inputs = {
			"enc_inps:0":tf.placeholder(tf.string, shape=(None, self.conf.input_max_len), name="enc_inps"),
			"enc_lens:0":tf.placeholder(tf.int32, shape=[None], name="enc_lens")
		}
		# inputs for training period 
		inputs["dec_inps:0"] = tf.placeholder(tf.string, shape=[None, self.conf.output_max_len + 2], name="dec_inps")
		inputs["dec_lens:0"] = tf.placeholder(tf.int32, shape=[None], name="dec_lens")
		inputs["down_wgts:0"] = tf.placeholder(tf.float32, shape=[None], name="down_wgts")
		return inputs

	def build(self, inputs, for_deploy):
		scope = ""
		conf = self.conf
		name = self.name
		job_type = self.job_type
		dtype = self.dtype
		self.beam_splits = conf.beam_splits
		self.beam_size = 1 if not for_deploy else sum(self.beam_splits)

		self.enc_str_inps = inputs["enc_inps:0"]
		self.dec_str_inps = inputs["dec_inps:0"]
		self.enc_lens = inputs["enc_lens:0"] 
		self.dec_lens = inputs["dec_lens:0"]
		self.down_wgts = inputs["down_wgts:0"]

		with tf.name_scope("TableLookup"):
			# Input maps
			self.in_table = lookup.MutableHashTable(key_dtype=tf.string,
														 value_dtype=tf.int64,
														 default_value=UNK_ID,
														 shared_name="in_table",
														 name="in_table",
														 checkpoint=True)

			self.out_table = lookup.MutableHashTable(key_dtype=tf.int64,
													 value_dtype=tf.string,
													 default_value="_UNK",
													 shared_name="out_table",
													 name="out_table",
													 checkpoint=True)
			# lookup
			self.enc_inps = self.in_table.lookup(self.enc_str_inps)
			self.dec_inps = self.in_table.lookup(self.dec_str_inps)

		graphlg.info("Preparing decoder inps...")
		dec_inps = tf.slice(self.dec_inps, [0, 0], [-1, conf.output_max_len + 1])

		# Create encode graph and get attn states
		graphlg.info("Creating embeddings and embedding enc_inps.")
		with ops.device("/cpu:0"):
			self.embedding = variable_scope.get_variable("embedding", [conf.output_vocab_size, conf.embedding_size])
		with tf.name_scope("Embed") as scope:
			dec_inps = tf.slice(self.dec_inps, [0, 0], [-1, conf.output_max_len + 1])
			with ops.device("/cpu:0"):
				self.emb_inps = embedding_lookup_unique(self.embedding, self.enc_inps)
				emb_dec_inps = embedding_lookup_unique(self.embedding, dec_inps)

		graphlg.info("Creating dynamic x rnn...")
		self.enc_outs, self.enc_states, mem_size, enc_state_size = DynRNN(conf.cell_model, conf.num_units, conf.num_layers,
																		self.emb_inps, self.enc_lens, keep_prob=1.0,
																		bidi=conf.bidirectional, name_scope="DynRNNEncoder")
		batch_size = tf.shape(self.enc_outs)[0]

		if self.conf.attention:
			init_h = self.enc_states[-1].h
		else:
			mechanism = dynamic_attention_wrapper.LuongAttention(num_units=conf.num_units, memory=self.enc_outs, 
																	max_mem_size=self.conf.input_max_len,
																	memory_sequence_length=self.enc_lens)
			init_h = mechanism(self.enc_states[-1].h)

		if isinstance(self.enc_states[-1], LSTMStateTuple):
			enc_state = LSTMStateTuple(self.enc_states[-1].c, init_h) 
			all_emb = tf.concat([enc_state.c, enc_state.h], 1) 
		else:
			all_emb = enc_state

		all_emb = tf.Print(all_emb, [tf.shape(all_emb)[0]], message="batch_size")

		query_emb, can_embs = tf.split(all_emb, [1, -1], 0)
		query_emb_normalized = tf.nn.l2_normalize(query_emb, 1)
		can_embs_normalized = tf.nn.l2_normalize(can_embs, 1)
		cos_dist_embs = tf.reduce_sum(query_emb_normalized * can_embs_normalized, 1)

		sum_word_embs = tf.reduce_sum(self.emb_inps, 1)
		query_word_emb, can_word_embs = tf.split(sum_word_embs, [1, -1], 0)	
		query_word_emb_normalized = tf.nn.l2_normalize(query_word_emb, 1)
		can_word_embs_normalized = tf.nn.l2_normalize(can_word_embs, 1)
		cos_dist_word_embs = tf.reduce_sum(query_word_emb_normalized * can_word_embs_normalized, 1)
		
		inputs = { 
			"enc_inps:0":self.enc_str_inps,
			"enc_lens:0":self.enc_lens
		}

		graph_nodes = {
			"loss":None,
			"inputs":inputs,
			"outputs":{"rnn_enc":tf.concat([tf.zeros([1]), cos_dist_embs], 0), "sum_emb":tf.concat([tf.zeros([1]), cos_dist_word_embs], 0)},
		}
		return graph_nodes

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
			k = k[0:self.conf.output_vocab_size]
			v = [i for i in range(len(k))]
			op_in = self.in_table.insert(constant_op.constant(k), constant_op.constant(v, dtype=tf.int64))
			op_out = self.out_table.insert(constant_op.constant(v,dtype=tf.int64), constant_op.constant(k))
			init_ops.extend([op_in, op_out])
		return init_ops

	def get_restorer(self):
		var_list = self.global_params + self.trainable_params + self.optimizer_params + tf.get_default_graph().get_collection("saveable_objects")

		## Just for the FUCKING naming compatibility to tensorflow 1.1
		var_map = {}
		for each in var_list:
			name = each.name
			#name = re.sub("lstm_cell/bias", "lstm_cell/biases", name)
			#name = re.sub("lstm_cell/kernel", "lstm_cell/weights", name)
			#name = re.sub("gates/bias", "gates/biases", name)
			#name = re.sub("candidate/bias", "candidate/biases", name)
			#name = re.sub("gates/kernel", "gates/weights", name)
			#name = re.sub("candidate/kernel", "candidate/weights", name)
			#name = re.sub("bias", "biases", name)
			#name = re.sub("dense/weights", "dense/kernel", name)
			#name = re.sub("dense/biases", "dense/bias", name)
			name = re.sub(":0", "", name)
			var_map[name] = each

		restorer = tf.train.Saver(var_list=var_map)
		return restorer

	def after_proc(self, out):
		keys = out.keys()
		values = out.values()
		after_proc_out = []
		for n, v in enumerate(zip(*values)):
			if n == 0:
				continue
			pair = {k:str(v[i]) for i, k in enumerate(keys)}
			pair["model_name"] = self.name
			after_proc_out.append(pair)
		return after_proc_out
