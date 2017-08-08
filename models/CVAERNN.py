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

import Nick_plan
import logging as log

graphlg = log.getLogger("graph")
DynamicAttentionWrapper = dynamic_attention_wrapper.DynamicAttentionWrapper
DynamicAttentionWrapperState = dynamic_attention_wrapper.DynamicAttentionWrapperState 
Bahdanau = dynamic_attention_wrapper.BahdanauAttention
Luong = dynamic_attention_wrapper.LuongAttention

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

def PriorNet(state, hidden_units, enc_latent_dim, stddev=1.0, prior_type="mlp"):
	epsilon = tf.random_normal([tf.shape(state)[0], enc_latent_dim], name="epsilon", stddev=stddev)
	if prior_type == "mlp":
		mu_layer1 = layers_core.Dense(hidden_units, use_bias=True, name="mu_layer1", activation=None)
		mu_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="mu_layer2")
		logvar_layer1 = layers_core.Dense(hidden_units, use_bias=True, name="logvar_layer1", activation=None)
		logvar_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="logvar_layer2")
		mu_prior = mu_layer2(mu_layer1(state))
		logvar_prior = logvar_layer2(logvar_layer1(state))
		z = mu_prior + tf.exp(0.5 * logvar_prior) * epsilon
	elif prior_type == "simple":
		mu_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="mu_layer2", activation=tf.tanh)
		logvar_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="logvar_layer2", activation=tf.tanh)
		mu_prior = mu_layer2(state)
		logvar_prior = logvar_layer2(state)
		z = mu_prior + tf.exp(0.5 * logvar_prior) * epsilon
	else:
		z = epsilon
		mu_prior = 0
		logvar_prior = 2 * tf.log(stddev)
	return z, mu_prior, logvar_prior 

def CreateVAE(states, enc_latent_dim, mu_prior, logvar_prior, stddev=1.0, name="vae", reuse=False, dtype=tf.float32):
	with variable_scope.variable_scope(name, dtype=dtype, reuse=reuse) as scope: 
		l2_loss = tf.constant(0.0)	 
		graphlg.info("Creating latent z for encoder states") 
		epsilon = tf.random_normal([tf.shape(states)[0], enc_latent_dim], name="epsilon", stddev=stddev)
		if states == None:
			graphlg.error("None states !!!") 
			exit(0)
		h_state = states
		# miu and sigma param for conditional latent variables
		W_enc_hidden_mu = tf.get_variable(name="w_enc_hidden_mu", shape=[int(h_state.shape[1]), enc_latent_dim], dtype=tf.float32,
											initializer=None)
		b_enc_hidden_mu = tf.get_variable(name="b_enc_hidden_mu", shape=[enc_latent_dim], dtype=tf.float32,
											initializer=None) 
		W_enc_hidden_logvar = tf.get_variable(name="w_enc_hidden_logvar", shape=[int(h_state.shape[1]), enc_latent_dim], dtype=tf.float32,
												initializer=None)
		b_enc_hidden_logvar = tf.get_variable(name="b_enc_hidden_logvar", shape=[enc_latent_dim], dtype=tf.float32,
												initializer=None) 
		# Should there be any non-linearty?
		mu_enc = tf.matmul(h_state, W_enc_hidden_mu) + b_enc_hidden_mu
		logvar_enc = tf.matmul(h_state, W_enc_hidden_logvar) + b_enc_hidden_logvar

		# regular
		l2_loss += tf.nn.l2_loss(W_enc_hidden_mu)
		l2_loss += tf.nn.l2_loss(W_enc_hidden_logvar)
		z = mu_enc + tf.exp(0.5 * logvar_enc) * epsilon

		# KL divergence
		# log(sigma2^2/sigma1^2) + (sigma1^2 + (mu1 - mu2)^2) / (2 * sigma2^2) - 0.5
		KLD = -0.5 * tf.reduce_sum(1 + logvar_enc - logvar_prior - (tf.pow(mu_enc - mu_prior, 2) + tf.exp(logvar_enc))/tf.exp(logvar_prior), axis = 1)
		return z, KLD, l2_loss

def DynRNN(cell_model, num_units, num_layers, emb_inps, enc_lens, keep_prob=1.0, bidi=False, name_scope="encoder", dtype=tf.float32):
	if bidi:
		with variable_scope.variable_scope(name_scope, dtype=dtype) as scope: 
			cell_fw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob)
			cell_bw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob)
		enc_outs, enc_states = bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
														inputs=emb_inps,
														sequence_length=enc_lens,
														dtype=dtype,
														parallel_iterations=16,
														scope=scope)

		fw_s, bw_s = enc_states 
		enc_states = []
		for f, b in zip(fw_s, bw_s):
			if isinstance(f, LSTMStateTuple):
				enc_states.append(LSTMStateTuple(tf.concat([f.c, b.c], axis=1), tf.concat([f.h, b.h], axis=1)))
			else:
				enc_states.append(tf.concat([f, b], 1))

		enc_outs = tf.concat([enc_outs[0], enc_outs[1]], axis=2)
		mem_size = 2 * num_units
		enc_state_size = 2 * num_units 
	else:
		with variable_scope.variable_scope(name_scope, dtype=dtype) as scope: 
			cell = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob)
		enc_outs, enc_states = dynamic_rnn(cell=cell,
										   inputs=emb_inps,
										   sequence_length=enc_lens,
										   parallel_iterations=16,
										   scope=scope,
										   dtype=dtype)
		mem_size = num_units
		enc_state_size = num_units
	return enc_outs, enc_states, mem_size, enc_state_size 

def AttnCell(cell_model, num_units, num_layers, memory, mem_lens, attn_type, max_mem_size, keep_prob=1.0, addmem=False, dtype=tf.float32):
	# Attention  
	with variable_scope.variable_scope("decoder", dtype=dtype) as scope: 
		decoder_cell = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob)

	if attn_type == "Luo":
		mechanism = dynamic_attention_wrapper.LuongAttention(num_units=num_units, memory=memory, max_mem_size=max_mem_size,
																memory_sequence_length=mem_lens)
	elif attn_type == "Bah":
		mechanism = dynamic_attention_wrapper.BahdanauAttention(num_units=num_units, memory=memory, max_mem_size=max_mem_size,
																memory_sequence_length=mem_lens)
	elif attn_type == None:
		return decoder_cell
	else:
		print "Unknown attention stype, must be Luo or Bah" 
		exit(0)
	attn_cell = DynamicAttentionWrapper(cell=decoder_cell, attention_mechanism=mechanism,
											attention_size=num_units, addmem=addmem)
	return attn_cell




def DecStateInit(thinking_state, num_units, decoder_cell, batch_size, beam_size):
	# Encoder states for initial state, with vae 
	zero_states = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_size)

	if isinstance(zero_states, DynamicAttentionWrapperState):
		cell_states = zero_states.cell_state
	else:
		cell_states = zero_states

	init_states = []
	for i, each in enumerate(cell_states):
	  if i > 0:
		init_states.append(each)
	  elif isinstance(each, LSTMStateTuple):
		new_c = tf.reshape(tf.concat([each.c], 1), [-1, num_units])
		new_h = tf.reshape(tf.concat([thinking_state] * beam_size, 1), [-1, num_units])
		init_states.append(LSTMStateTuple(new_c, new_h))
	  else:
		state = tf.reshape(tf.concat([thinking_state] * beam_size, 1), [-1, num_units])
		init_states.append(state)

	if isinstance(decoder_cell,DynamicAttentionWrapper):
		zero_states = DynamicAttentionWrapperState(tuple(init_states), zero_states.attention, zero_states.newmem, zero_states.alignments)
	else:
		zero_states = tuple(init_states)

	return zero_states



class CVAERNN(ModelCore):
	def __init__(self, name, job_type="single", task_id=0, dtype=tf.float32):
		super(self.__class__, self).__init__(name, job_type, task_id, dtype) 
		self.embedding = None
		self.out_proj = None

	def build(self, for_deploy, variants=""):
		scope = ""
		conf = self.conf
		name = self.name
		job_type = self.job_type
		dtype = self.dtype
		self.beam_splits = conf.beam_splits
		self.beam_size = 1 if not for_deploy else sum(self.beam_splits)

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
		graphlg.info("Creating placeholders...")
		self.enc_str_inps = tf.placeholder(tf.string, shape=(None, conf.input_max_len), name="enc_inps") 
		self.enc_lens = tf.placeholder(tf.int32, shape=[None], name="enc_lens") 
		# inputs for training period 
		self.dec_str_inps = tf.placeholder(tf.string, shape=[None, conf.output_max_len + 2], name="dec_inps") 
		self.dec_lens = tf.placeholder(tf.int32, shape=[None], name="dec_lens") 
		self.down_wgts = tf.placeholder(tf.float32, shape=[None], name="down_wgts")

		# lookup
		self.enc_inps = self.in_table.lookup(self.enc_str_inps)
		self.dec_inps = self.in_table.lookup(self.dec_str_inps)

		graphlg.info("Preparing decoder inps...")
		dec_inps = tf.slice(self.dec_inps, [0, 0], [-1, conf.output_max_len + 1])

		# Create encode graph and get attn states
		graphlg.info("Creating embeddings and embedding enc_inps.")

		with ops.device("/cpu:0"):
			self.embedding = variable_scope.get_variable("embedding", [conf.output_vocab_size, conf.embedding_size])
			self.emb_inps = embedding_lookup_unique(self.embedding, self.enc_inps)
			emb_dec_inps = embedding_lookup_unique(self.embedding, dec_inps)

		graphlg.info("Creating dynamic x rnn...")
		self.enc_outs, self.enc_states, mem_size, enc_state_size = DynRNN(conf.cell_model, conf.num_units, conf.num_layers,
																		self.emb_inps, self.enc_lens, keep_prob=1.0, bidi=False, name_scope="x_enc")
		memory = tf.reshape(tf.concat([self.enc_outs] * self.beam_size, 2), [-1, conf.input_max_len, mem_size])
		memory_lens = tf.squeeze(tf.reshape(tf.concat([tf.expand_dims(self.enc_lens, 1)] * self.beam_size, 1), [-1, 1]), 1)
		batch_size = tf.shape(self.enc_outs)[0]

		x_s = self.enc_states[-1].h if isinstance(self.enc_states[-1], LSTMStateTuple) else self.enc_states[-1]

		# Not sure, its's only for the gradient bug fixup 
		x_s = x_s + tf.reduce_sum(self.enc_outs, 1)
		KLD = 0.0
		hidden_units = int(math.sqrt(mem_size * self.conf.enc_latent_dim))

		# Different graph for training and inference time 
		if not for_deploy:
			# Y inputs for posterior z 
			y_emb_inps = tf.slice(emb_dec_inps, [0, 1, 0], [-1, -1, -1])
			y_enc_outs, y_enc_states, y_mem_size, y_enc_state_size = DynRNN(conf.cell_model, conf.num_units, conf.num_layers,
																		y_emb_inps, self.dec_lens, keep_prob=1.0, bidi=False, name_scope="y_enc")
			#y_memory = tf.reshape(tf.concat([self.enc_outs] * self.beam_size, 2), [-1, conf.input_max_len, mem_size])
			#y_memory_lens = tf.squeeze(tf.reshape(tf.concat([tf.expand_dims(self.enc_lens, 1)] * self.beam_size, 1), [-1, 1]), 1)
			y_s = y_enc_states[-1].h if isinstance(y_enc_states[-1], LSTMStateTuple) else y_enc_states[-1]

			enc_layer = layers_core.Dense(mem_size, use_bias=True, name="enc_x_y", activation=tf.tanh)
			x_y_enc = enc_layer(tf.concat([x_s, y_s], 1))

			_, mu_prior, logvar_prior = PriorNet(x_s, hidden_units, self.conf.enc_latent_dim, stddev=1.0, prior_type=conf.prior_type)
			z, KLD, l2 = CreateVAE(x_y_enc, self.conf.enc_latent_dim, mu_prior, logvar_prior, stddev=self.conf.stddev)
		else:
			z, _, _ = PriorNet(x_s, hidden_units, self.conf.enc_latent_dim, stddev=1.0, prior_type=conf.prior_type)
				
		# thinking states
		thinking_state = tf.concat([z, x_s], 1)

		# project z + x_thinking_state to decoder state
		decision_layer = layers_core.Dense(mem_size, use_bias=True, name="decide", activation=tf.tanh)
		decision_state = decision_layer(thinking_state) 

		# add BOW loss
		num_hidden_units = int(math.sqrt(conf.output_vocab_size * int(decision_state.shape[1])))
		bow_l1 = layers_core.Dense(num_hidden_units, use_bias=True, name="bow_hidden", activation=tf.tanh)
		bow_l2 = layers_core.Dense(conf.output_vocab_size, use_bias=True, name="bow_out", activation=tf.tanh)
		bow = bow_l2(bow_l1(decision_state)) 

		y_dec_inps = tf.slice(self.dec_inps, [0, 1], [-1, -1])
		bow_y = tf.reduce_sum(tf.one_hot(y_dec_inps, on_value=1.0, off_value=0.0, axis=-1, depth=conf.output_vocab_size), axis=1)
		batch_bow_losses = tf.reduce_sum(bow_y * (-1.0) * tf.nn.log_softmax(bow), axis=1)

		max_mem_size = self.conf.input_max_len + self.conf.output_max_len + 2
		cell = AttnCell(cell_model=conf.cell_model, num_units=mem_size, num_layers=conf.num_layers, attn_type=self.conf.attention,
						memory=memory, mem_lens=memory_lens, max_mem_size=max_mem_size, addmem=self.conf.addmem,
						keep_prob=1.0, dtype=tf.float32)

		# Fit decision states to shape of attention decoder cell states 
		zero_attn_states = DecStateInit(decision_state, mem_size, cell, batch_size, self.beam_size)
		
		# Output projection
		graphlg.info("Creating out_proj...") 
		if conf.out_layer_size:
			w = tf.get_variable("proj_w", [conf.out_layer_size, conf.output_vocab_size], dtype=dtype)
		else:
			w = tf.get_variable("proj_w", [mem_size, conf.output_vocab_size], dtype=dtype)
		b = tf.get_variable("proj_b", [conf.output_vocab_size], dtype=dtype)
		self.out_proj = (w, b)

		if not for_deploy: 
			inputs = {}
			dec_init_state = zero_attn_states
			hp_train = helper.ScheduledEmbeddingTrainingHelper(inputs=emb_dec_inps, sequence_length=self.dec_lens, 
															   embedding=self.embedding, sampling_probability=0.0,
															   out_proj=self.out_proj)
			output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
			my_decoder = basic_decoder.BasicDecoder(cell=cell, helper=hp_train, initial_state=dec_init_state, output_layer=output_layer)
			cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, impute_finished=(self.conf.addmem==False),
															maximum_iterations=conf.output_max_len + 1, scope=scope)
			outputs = cell_outs.rnn_output

			L = tf.shape(outputs)[1]
			outputs = tf.reshape(outputs, [-1, int(self.out_proj[0].shape[0])])
			outputs = tf.matmul(outputs, self.out_proj[0]) + self.out_proj[1] 
			logits = tf.reshape(outputs, [-1, L, int(self.out_proj[0].shape[1])])

			# branch 1 for debugging, doesn't have to be called
			#m = tf.shape(self.outputs)[0]
			#self.mask = tf.zeros([m, int(w.shape[1])])
			#for i in [3]:
			#	self.mask = self.mask + tf.one_hot(indices=tf.ones([m], dtype=tf.int32) * i, on_value=100.0, depth=int(w.shape[1]))
			#self.outputs = self.outputs - self.mask

			self.outputs = tf.argmax(logits, axis=2)
			self.outputs = tf.reshape(self.outputs, [-1, L])
			self.outputs = self.out_table.lookup(tf.cast(self.outputs, tf.int64))

			# branch 2 for loss
			tars = tf.slice(self.dec_inps, [0, 1], [-1, L])
			wgts = tf.cumsum(tf.one_hot(self.dec_lens, L), axis=1, reverse=True)

			#wgts = wgts * tf.expand_dims(self.down_wgts, 1)
			self.loss = loss.sequence_loss(logits=logits, targets=tars, weights=wgts, average_across_timesteps=False, average_across_batch=False)
			batch_wgt = tf.reduce_sum(self.down_wgts) + 1e-12 
			bow_loss = tf.reduce_sum(batch_bow_losses * self.down_wgts) / batch_wgt

			example_losses = tf.reduce_sum(self.loss, 1)
			see_loss = tf.reduce_sum(example_losses / tf.cast(self.dec_lens, tf.float32) * self.down_wgts) / batch_wgt
			KLD = tf.reduce_sum(KLD * self.down_wgts) / batch_wgt

			self.loss = tf.reduce_sum(example_losses * self.down_wgts) / batch_wgt
			tf.summary.scalar("loss", see_loss)

			if conf.kld_ratio:
				self.loss += conf.kld_ratio * KLD
				tf.summary.scalar("kld", KLD) 
			if conf.bow_ratio:
				self.loss += conf.bow_ratio * bow_loss 
				tf.summary.scalar("bow", bow_loss)

			graph_nodes = {
				"loss":self.loss,
				"inputs":inputs,
				"debug_outputs":self.outputs,
				"outputs":None
				"visualize":None
			}
			return graph_nodes
		else:
			hp_infer = helper.GreedyEmbeddingHelper(embedding=self.embedding,
													start_tokens=tf.ones(shape=[batch_size * self.beam_size], dtype=tf.int32),
													end_token=EOS_ID, out_proj=self.out_proj)
			output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
			dec_init_state = beam_decoder.BeamState(tf.zeros([batch_size * self.beam_size]), zero_attn_states, tf.zeros([batch_size * self.beam_size], tf.int32))

			my_decoder = beam_decoder.BeamDecoder(cell=cell, helper=hp_infer, out_proj=self.out_proj, initial_state=dec_init_state,
													beam_splits=self.beam_splits, max_res_num=self.conf.max_res_num, output_layer=output_layer)
			cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, scope=scope, maximum_iterations=self.conf.output_max_len)

			L = tf.shape(cell_outs.beam_ends)[1]
			beam_symbols = cell_outs.beam_symbols
			beam_parents = cell_outs.beam_parents

			beam_ends = cell_outs.beam_ends
			beam_end_parents = cell_outs.beam_end_parents
			beam_end_probs = cell_outs.beam_end_probs
			alignments = cell_outs.alignments

			beam_ends = tf.reshape(tf.transpose(beam_ends, [0, 2, 1]), [-1, L])
			beam_end_parents = tf.reshape(tf.transpose(beam_end_parents, [0, 2, 1]), [-1, L])
			beam_end_probs = tf.reshape(tf.transpose(beam_end_probs, [0, 2, 1]), [-1, L])


			# Creating tail_ids 
			batch_size = tf.Print(batch_size, [batch_size], message="CVAERNN batch")

			#beam_symbols = tf.Print(cell_outs.beam_symbols, [tf.shape(cell_outs.beam_symbols)], message="beam_symbols")
			#beam_parents = tf.Print(cell_outs.beam_parents, [tf.shape(cell_outs.beam_parents)], message="beam_parents")
			#beam_ends = tf.Print(cell_outs.beam_ends, [tf.shape(cell_outs.beam_ends)], message="beam_ends") 
			#beam_end_parents = tf.Print(cell_outs.beam_end_parents, [tf.shape(cell_outs.beam_end_parents)], message="beam_end_parents") 
			#beam_end_probs = tf.Print(cell_outs.beam_end_probs, [tf.shape(cell_outs.beam_end_probs)], message="beam_end_probs") 
			#alignments = tf.Print(cell_outs.alignments, [tf.shape(cell_outs.alignments)], message="beam_attns")

			batch_offset = tf.expand_dims(tf.cumsum(tf.ones([batch_size, self.beam_size], dtype=tf.int32) * self.beam_size, axis=0, exclusive=True), 2)
			offset2 = tf.expand_dims(tf.cumsum(tf.ones([batch_size, self.beam_size * 2], dtype=tf.int32) * self.beam_size, axis=0, exclusive=True), 2)

			out_len = tf.shape(beam_symbols)[1]
			self.beam_symbol_strs = tf.reshape(self.out_table.lookup(tf.cast(beam_symbols, tf.int64)), [batch_size, self.beam_size, -1])
			self.beam_parents = tf.reshape(beam_parents, [batch_size, self.beam_size, -1]) - batch_offset

			self.beam_ends = tf.reshape(beam_ends, [batch_size, self.beam_size * 2, -1])
			self.beam_end_parents = tf.reshape(beam_end_parents, [batch_size, self.beam_size * 2, -1]) - offset2
			self.beam_end_probs = tf.reshape(beam_end_probs, [batch_size, self.beam_size * 2, -1])
			self.beam_attns = tf.reshape(alignments, [batch_size, self.beam_size, out_len, -1])

			#cell_outs.alignments
			#self.outputs = tf.concat([outputs_str, tf.cast(cell_outs.beam_parents, tf.string)], 1)

			#ones = tf.ones([batch_size, self.beam_size], dtype=tf.int32)
			#aux_matrix = tf.cumsum(ones * self.beam_size, axis=0, exclusive=True)

			#tm_beam_parents_reverse = tf.reverse(tf.transpose(cell_outs.beam_parents), axis=[0])
			#beam_probs = final_state[1] 

			#def traceback(prev_out, curr_input):
			#	return tf.gather(curr_input, prev_out) 
			#	
			#tail_ids = tf.reshape(tf.cumsum(ones, axis=1, exclusive=True) + aux_matrix, [-1])
			#tm_symbol_index_reverse = tf.scan(traceback, tm_beam_parents_reverse, initializer=tail_ids)
			## Create beam index for symbols, and other info  
			#tm_symbol_index = tf.concat([tf.expand_dims(tail_ids, 0), tm_symbol_index_reverse], axis=0)
			#tm_symbol_index = tf.reverse(tm_symbol_index, axis=[0])
			#tm_symbol_index = tf.slice(tm_symbol_index, [1, 0], [-1, -1])
			#symbol_index = tf.expand_dims(tf.transpose(tm_symbol_index), axis=2)
			#symbol_index = tf.concat([symbol_index, tf.cumsum(tf.ones_like(symbol_index), exclusive=True, axis=1)], axis=2)

			## index alignments and output symbols
			#alignments = tf.gather_nd(cell_outs.alignments, symbol_index)
			#symbol_ids = tf.gather_nd(cell_outs.beam_symbols, symbol_index)

			## outputs and other info
			#self.others = [alignments, beam_probs]
			#self.outputs = self.out_table.lookup(tf.cast(symbol_ids, tf.int64))

			inputs = { 
				"enc_inps:0":self.enc_str_inps,
				"enc_lens:0":self.enc_lens
			}
			outputs = {
				"beam_symbols":self.beam_symbol_strs,
				"beam_parents":self.beam_parents,
				"beam_ends":self.beam_ends,
				"beam_end_parents":self.beam_end_parents,
				"beam_end_probs":self.beam_end_probs,
				"beam_attns":self.beam_attns
			}

			graph_nodes = {
				"loss":None,
				"inputs":inputs,
				"outputs":outputs,
				"visualize":{
					"thinking":thinking_state,
					"decision":decision_state
				}
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
			name = re.sub("lstm_cell/bias", "lstm_cell/biases", name)
			name = re.sub("lstm_cell/kernel", "lstm_cell/weights", name)
			name = re.sub("gates/bias", "gates/biases", name)
			name = re.sub("candidate/bias", "candidate/biases", name)
			name = re.sub("gates/kernel", "gates/weights", name)
			name = re.sub("candidate/kernel", "candidate/weights", name)
			#name = re.sub("bias", "biases", name)
			#name = re.sub("dense/weights", "dense/kernel", name)
			#name = re.sub("dense/biases", "dense/bias", name)
			name = re.sub(":0", "", name)
			var_map[name] = each

		restorer = tf.train.Saver(var_list=var_map)
		return restorer

	def after_proc(self, out):
		outputs, probs, attns = Nick_plan.handle_beam_out(out, self.conf.beam_splits)
		after_proc_out = {
			"outputs":outputs,
			"probs":probs,
			"attns":attns
		}
		return after_proc_out 

	def Project(self, session, records, tensor):
		#embedding = get_dtype=tensor.dtype, tensor_array_name="proj_name", size=len(records), infer_shape=False)
		emb_list = [] 
		out_list = []
		for start in range(0, len(records), self.conf.batch_size):
			batch = records[start:start + self.conf.batch_size]
			examples = self.fetch_test_data(batch, begin=0, size=len(batch))
			input_feed = self.get_batch(examples)
			a = session.run([tensor, self.outputs], feed_dict=input_feed)
			emb_list.append(a[0])
			out_list.append(a[1])
		embs = np.concatenate(emb_list,axis=0)
		outs = np.concatenate(out_list,axis=0)
		return embs, outs

if __name__ == "__main__":
	#name = "cvae4-merge-stc-weibo" 
	#name = "cvae4_1-weibo-stc-bought" 

	#somehow good
	name = "cvae-simpleprior-reddit-addmem"
	#name = "cvae-1024-simpleprior-attn"
	#name = "cvae-1024-simpleprior-attn-addmem"
	#name = "cvae-1024-prior-attn-addmem"

	#name = "cvae-512-prior-attn"
	#name = "cvae-512-prior-noattn"
	#name = "cvae-128-prior-attn"
	#name = "cvae-512-noprior-noattn"
	#name = "cvae-bi-simpleprior-attn-poem"
	model = CVAERNN(name)
	if len(sys.argv) == 2:
		gpu = 0
	flag = sys.argv[1]
	#model(flag)
	model(flag, use_seg=False)
