#from __future__ import absolute_import
import time
import numpy as np
from ModelCore import *

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.contrib import lookup

from tensorflow.contrib.rnn import  MultiRNNCell, AttentionCellWrapper, GRUCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.seq2seq.python.ops import loss
from tensorflow.python.layers import core as layers_core

import Nick_plan
import logging as log

graphlg = log.getLogger("graph")

DynamicAttentionWrapper = dynamic_attention_wrapper.DynamicAttentionWrapper
DynamicAttentionWrapperState = dynamic_attention_wrapper.DynamicAttentionWrapperState 
Bahdanau = dynamic_attention_wrapper.BahdanauAttention
Luong = dynamic_attention_wrapper.LuongAttention

def DynRNN(cell_model, num_units, num_layers, emb_inps, enc_lens, keep_prob=1.0, bidi=False, name_scope="encoder", dtype=tf.float32):
	with tf.name_scope(name_scope) as scope:
		if bidi:
			cell_fw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, name_scope="cell_fw")
			cell_bw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, name_scope="cell_bw")
			enc_outs, enc_states = bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
															inputs=emb_inps,
															sequence_length=enc_lens,
															dtype=dtype,
															parallel_iterations=16,
															scope=name_scope)
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
			cell = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, name_scope="cell")
			enc_outs, enc_states = dynamic_rnn(cell=cell,
											   inputs=emb_inps,
											   sequence_length=enc_lens,
											   parallel_iterations=16,
											   dtype=dtype,
											   scope=name_scope)
			mem_size = num_units
			enc_state_size = num_units
	return enc_outs, enc_states, mem_size, enc_state_size

def CreateMultiRNNCell(cell_name, num_units, num_layers=1, output_keep_prob=1.0, reuse=False, name_scope=None):
	with tf.variable_scope(name_scope):
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
			cells.append(single_cell)
	return MultiRNNCell(cells) 

class AttnSeq2Seq(ModelCore):
	"""
		standard attention seq2seq
	"""
	def __init__(self, name, job_type="single", task_id=0, dtype=tf.float32):
		super(self.__class__, self).__init__(name, job_type, task_id, dtype) 

	def build(self, for_deploy, variants=""):
		conf = self.conf
		name = self.name
		job_type = self.job_type
		dtype = self.dtype
		self.beam_size = 1 if (not for_deploy or variants=="score") else sum(self.conf.beam_splits)
		conf.keep_prob = conf.keep_prob if not for_deploy else 1.0

		
		graphlg.info("Creating placeholders...")
		self.enc_str_inps = tf.placeholder(tf.string, shape=(None, conf.input_max_len), name="enc_inps") 
		self.enc_lens = tf.placeholder(tf.int32, shape=[None], name="enc_lens") 
		self.dec_str_inps = tf.placeholder(tf.string, shape=[None, conf.output_max_len + 2], name="dec_inps") 
		self.dec_lens = tf.placeholder(tf.int32, shape=[None], name="dec_lens") 
		self.down_wgts = tf.placeholder(tf.float32, shape=[None], name="down_wgts")

		with tf.name_scope("TableLookup"):
			# lookup tables
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
			self.enc_inps = self.in_table.lookup(self.enc_str_inps)
			self.dec_inps = self.in_table.lookup(self.dec_str_inps)
		
		if for_deploy:
			self.enc_inps = tf.Print(self.enc_inps, [self.enc_inps], message="enc ids", summarize=10000)
			self.enc_lens = tf.Print(self.enc_lens, [self.enc_lens], message="enc lens", summarize=10000)

		# Create encode graph and get attn states
		graphlg.info("Creating embeddings and embedding enc_inps.")
		with ops.device("/cpu:0"):
			self.embedding = variable_scope.get_variable("embedding", [conf.output_vocab_size, conf.embedding_size])

		with tf.name_scope("Embed") as scope:
			dec_inps = tf.slice(self.dec_inps, [0, 0], [-1, conf.output_max_len + 1])
			with ops.device("/cpu:0"):
				self.emb_inps = embedding_lookup_unique(self.embedding, self.enc_inps)
				emb_dec_inps = embedding_lookup_unique(self.embedding, dec_inps)

		graphlg.info("Creating dynamic rnn...")
		self.enc_outs, self.enc_states, mem_size, enc_state_size = DynRNN(conf.cell_model, conf.num_units, conf.num_layers,
																			  self.emb_inps, self.enc_lens, keep_prob=self.conf.keep_prob,
																			  bidi=conf.bidirectional, name_scope="DynRNNEncoder")
		enc_state = self.enc_states[-1]

		with tf.name_scope("ShapeToDecInit"):
			final_enc_states = [enc_state]
			with tf.name_scope("ShapeToBeam") as scope: 
				memory = tf.reshape(tf.tile(self.enc_outs, [1, 1, self.beam_size]), [-1, conf.input_max_len, mem_size])
				memory_lens = tf.squeeze(tf.reshape(tf.tile(tf.expand_dims(self.enc_lens, 1), [1, self.beam_size]), [-1, 1]), 1)
				def _to_beam(t):
					return tf.reshape(tf.tile(t, [1, self.beam_size]), [-1, mem_size])	
				beam_init_states = tf.contrib.framework.nest.map_structure(_to_beam, final_enc_states)
				beam_zero_states = tf.contrib.framework.nest.map_structure(tf.zeros_like, beam_init_states) 

			# Construct init states for decoder
			init_states = []
			for i, each in enumerate(self.enc_states):
				if i == 0:
					init_states.extend(beam_init_states)
				else:
					init_states.extend(beam_zero_states)

		# output projector (w, b)
		with tf.variable_scope("OutProj"):
			if conf.out_layer_size:
				w = tf.get_variable("proj_w", [conf.out_layer_size, conf.output_vocab_size], dtype=dtype)
			elif conf.bidirectional:
				w = tf.get_variable("proj_w", [conf.num_units * 2, conf.output_vocab_size], dtype=dtype)
			else:
				w = tf.get_variable("proj_w", [conf.num_units, conf.output_vocab_size], dtype=dtype)
			b = tf.get_variable("proj_b", [conf.output_vocab_size], dtype=dtype)

		with tf.name_scope("DynRNNDecode") as scope:
			decoder_cell = CreateMultiRNNCell(conf.cell_model, mem_size, conf.num_layers, conf.output_keep_prob, name_scope=scope)
			max_mem_size = self.conf.input_max_len + self.conf.output_max_len + 2
			if conf.attention == "Luo":
				mechanism = dynamic_attention_wrapper.LuongAttention(num_units=mem_size, memory=memory, max_mem_size=max_mem_size,
																		memory_sequence_length=memory_lens)
			elif conf.attention == "Bah":
				mechanism = dynamic_attention_wrapper.BahdanauAttention(num_units=mem_size, memory=memory, max_mem_size=max_mem_size,
																		memory_sequence_length=memory_lens)
			else:
				print "Unknown attention stype, must be Luo or Bah" 
				exit(0)

			attn_cell = DynamicAttentionWrapper(cell=decoder_cell, attention_mechanism=mechanism,
											attention_size=mem_size, addmem=self.conf.addmem)
			
			with tf.name_scope("InitStates") as init_state_scope:
				batch_size = tf.shape(self.enc_outs)[0]
				zero_attn_states = attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size * self.beam_size)
				zero_attn_states = DynamicAttentionWrapperState(tuple(init_states), zero_attn_states.attention, zero_attn_states.newmem, zero_attn_states.alignments)
				if not for_deploy:
					dec_init_state = zero_attn_states
				else:
					dec_init_state = beam_decoder.BeamState(tf.zeros([batch_size * self.beam_size]), zero_attn_states, tf.zeros([batch_size * self.beam_size], tf.int32))
				graphlg.info("Creating out_proj...") 

			if not for_deploy: 
				hp_train = helper.ScheduledEmbeddingTrainingHelper(inputs=emb_dec_inps, sequence_length=self.dec_lens, 
																   embedding=self.embedding, sampling_probability=self.conf.sample_prob,
																   out_proj=(w, b))
				output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
				my_decoder = basic_decoder.BasicDecoder(cell=attn_cell, helper=hp_train, initial_state=dec_init_state, output_layer=output_layer)
				cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, impute_finished=True, maximum_iterations=conf.output_max_len + 1, scope=scope)
			elif variants == "score":
				dec_init_state = zero_attn_states
				hp_train = helper.ScheduledEmbeddingTrainingHelper(inputs=emb_dec_inps, sequence_length=self.dec_lens, embedding=self.embedding, sampling_probability=0.0,
																   out_proj=(w, b))
				output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
				my_decoder = score_decoder.ScoreDecoder(cell=attn_cell, helper=hp_train, out_proj=(w, b), initial_state=dec_init_state, output_layer=output_layer)
				cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, scope=scope, maximum_iterations=self.conf.output_max_len, impute_finished=False)
			else:
				hp_infer = helper.GreedyEmbeddingHelper(embedding=self.embedding,
														start_tokens=tf.ones(shape=[batch_size * self.beam_size], dtype=tf.int32),
														end_token=EOS_ID, out_proj=(w, b))

				output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
				my_decoder = beam_decoder.BeamDecoder(cell=attn_cell, helper=hp_infer, out_proj=(w, b), initial_state=dec_init_state,
														beam_splits=self.conf.beam_splits, max_res_num=self.conf.max_res_num, output_layer=output_layer)
				cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, scope=scope, maximum_iterations=self.conf.output_max_len, impute_finished=True)

		if not for_deploy:	
			outputs = cell_outs.rnn_output
			# Output ouputprojected to logits
			L = tf.shape(outputs)[1]
			outputs = tf.reshape(outputs, [-1, int(w.shape[0])])
			outputs = tf.matmul(outputs, w) + b 
			logits = tf.reshape(outputs, [-1, L, int(w.shape[1])])

			# branch 1 for debugging, doesn't have to be called
			with tf.name_scope("DebugOutputs") as scope:
				self.outputs = tf.argmax(logits, axis=2)
				self.outputs = tf.reshape(self.outputs, [-1, L])
				self.outputs = self.out_table.lookup(tf.cast(self.outputs, tf.int64))

			with tf.name_scope("Loss") as scope:
				tars = tf.slice(self.dec_inps, [0, 1], [-1, L])
				wgts = tf.cumsum(tf.one_hot(self.dec_lens, L), axis=1, reverse=True)

				#wgts = wgts * tf.expand_dims(self.down_wgts, 1)
				self.loss = loss.sequence_loss(logits=logits, targets=tars, weights=wgts, average_across_timesteps=False, average_across_batch=False)
				example_losses = tf.reduce_sum(self.loss, 1)

				batch_wgt = tf.reduce_sum(self.down_wgts)
				see_loss = self.loss = tf.reduce_sum(example_losses / tf.cast(self.dec_lens, tf.float32) * self.down_wgts) / batch_wgt

			with tf.name_scope(self.model_kind):
				tf.summary.scalar("loss", see_loss)

			graph_nodes = {
				"loss":self.loss,
				"inputs":{},
				"outputs":{},
				"debug_outputs":self.outputs
			}

		elif variants == "score":	
			L = tf.shape(cell_outs.logprobs)[1]
			one_hot = tf.one_hot(tf.slice(self.dec_inps, [0, 1], [-1, L]), depth=self.conf.output_vocab_size, axis=-1, on_value=1.0, off_value=0.0)
			outputs = tf.reduce_sum(cell_outs.logprobs * one_hot, 2)
			outputs = tf.reduce_sum(outputs, axis=1)
			inputs = { 
				"enc_inps:0":self.enc_str_inps,
				"enc_lens:0":self.enc_lens,
				"dec_inps:0":self.dec_str_inps,
				"dec_lens:0":self.dec_lens
			}
			graph_nodes = {
				"loss":None,
				"inputs":inputs,
				"outputs":{"logprobs":outputs},
				"visualize":None
			}
		else:
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

			## Creating tail_ids 
			batch_size = tf.Print(batch_size, [batch_size], message="batch size")
			batch_offset = tf.expand_dims(tf.cumsum(tf.ones([batch_size, self.beam_size], dtype=tf.int32) * self.beam_size, axis=0, exclusive=True), 2)
			offset2 = tf.expand_dims(tf.cumsum(tf.ones([batch_size, self.beam_size * 2], dtype=tf.int32) * self.beam_size, axis=0, exclusive=True), 2)

			out_len = tf.shape(beam_symbols)[1]
			self.beam_symbol_strs = tf.reshape(self.out_table.lookup(tf.cast(beam_symbols, tf.int64)), [batch_size, self.beam_size, -1])
			self.beam_parents = tf.reshape(beam_parents, [batch_size, self.beam_size, -1]) - batch_offset

			self.beam_ends = tf.reshape(beam_ends, [batch_size, self.beam_size * 2, -1])
			self.beam_end_parents = tf.reshape(beam_end_parents, [batch_size, self.beam_size * 2, -1]) - offset2
			self.beam_end_probs = tf.reshape(beam_end_probs, [batch_size, self.beam_size * 2, -1])
			self.beam_attns = tf.reshape(alignments, [batch_size, self.beam_size, out_len, -1])
			
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
				"visualize":{}
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
			#name = re.sub("gru_cell/bias", "gru_cell/biases", name)
			#name = re.sub("gru_cell/kernel", "gru_cell/weights", name)
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
		outputs, probs, attns = Nick_plan.handle_beam_out(out, self.conf.beam_splits)

		outs = [[(outputs[n][i], probs[n][i]) for i in range(len(outputs[n]))] for n in range(len(outputs))]

		#sorted_outs = sorted(outs, key=lambda x:x[1]/len(x[0]), reverse=True)
		sorted_outs = [sorted(outs[n], key=lambda x:x[1], reverse=True) for n in range(len(outs))]
		after_proc_out = [[{"outputs":res[0], "probs":res[1]} for res in example] for example in sorted_outs]
		return after_proc_out 
