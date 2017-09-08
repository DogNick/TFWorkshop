#from __future__ import absolute_import
import sys
import time
import math
import numpy as np
import tensorflow as tf
from ModelCore import *


from tensorflow.contrib import lookup
from tensorflow.contrib.framework import nest
from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMStateTuple

from tensorflow.contrib.seq2seq import tile_batch 
from tensorflow.contrib.seq2seq.python.ops import loss

from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import core as layers_core

from components import * 

import Nick_plan
import logging as log

graphlg = log.getLogger("graph")

# 1, use new model class structure
# 2, use maybe attention for enc_states
# 3, put more preproc into tensorflow graph

class CVAEattnRNN2(ModelCore):
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
		dtype = self.dtype
		beam_size = 1 if not for_deploy else sum(conf.beam_splits)

		with tf.name_scope("WordEmbedding"):
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
			enc_inps = self.in_table.lookup(inputs["enc_inps:0"])
			dec_inps = self.in_table.lookup(inputs["dec_inps:0"])

			graphlg.info("Creating embeddings and embedding enc_inps.")
			with tf.device("/cpu:0"):
				self.embedding = variable_scope.get_variable("embedding", [conf.output_vocab_size, conf.embedding_size])
				emb_inps = embedding_lookup_unique(self.embedding, enc_inps)
				emb_dec_inps = embedding_lookup_unique(self.embedding, dec_inps)
			emb_dec_next_inps = tf.slice(emb_dec_inps, [0, 0, 0], [-1, conf.output_max_len + 1, -1])

		
		batch_size = tf.shape(enc_inps)[0]

		# Create encode graph and get attn states
		graphlg.info("Creating dynamic x rnn...")
		enc_outs, enc_states, mem_size, enc_state_size = DynEncode(conf.cell_model, conf.num_units, conf.num_layers,
																emb_inps, inputs["enc_lens:0"], keep_prob=1.0,
																bidi=conf.bidirectional, name_scope="DynEncodeX")
		
		with tf.variable_scope("AttnEncState") as scope2:
			mechanism = Luong1_2(num_units=conf.num_units, memory=enc_outs, max_mem_size=conf.input_max_len, memory_sequence_length=inputs["enc_lens:0"], name=scope2.original_name_scope)
			if isinstance(enc_states[-1], LSTMStateTuple):
				#score = tf.expand_dims(tf.nn.softmax(mechanism(enc_states[-1].h)), 1)
				score = tf.expand_dims(mechanism(enc_states[-1].h, ()), 1)
				attention_h = tf.squeeze(tf.matmul(score, enc_outs), 1)
				enc_state = LSTMStateTuple(enc_states[-1].c, attention_h) 
			else:
				#score = tf.expand_dims(tf.nn.softmax(mechanism(enc_states[-1])), 1)
				score = tf.expand_dims(mechanism(enc_states[-1], ()), 1)
				enc_state = tf.squeeze(tf.matmul(score, enc_outs), 1)

		hidden_units = int(math.sqrt(mem_size * conf.enc_latent_dim))
		z, mu_prior, logvar_prior = Ptheta([enc_state], hidden_units, conf.enc_latent_dim, stddev=1, prior_type=conf.prior_type, name_scope="EncToPtheta")

		KLD = 0.0
		# Y inputs for posterior z when training
		if not for_deploy:
			#with tf.name_scope("variational_distribution") as scope:
			y_emb_inps = tf.slice(emb_dec_inps, [0, 1, 0], [-1, -1, -1])
			y_enc_outs, y_enc_states, y_mem_size, y_enc_state_size = DynEncode(conf.cell_model, conf.num_units, conf.num_layers, y_emb_inps, inputs["dec_lens:0"],
																					keep_prob=conf.keep_prob, bidi=False, name_scope="DynEncodeY")
			z, KLD, l2 = VAE([enc_state, y_enc_states[-1]], conf.enc_latent_dim, mu_prior, logvar_prior, name_scope="VAE")

		# project z + x_thinking_state to decoder state
		with tf.name_scope("GatedZState"):
			if isinstance(enc_state, LSTMStateTuple):
				h_gate = tf.layers.dense(z, int(enc_state.h.get_shape()[1]), use_bias=True, name="z_gate_h", activation=tf.sigmoid)
				c_gate = tf.layers.dense(z, int(enc_state.c.get_shape()[1]), use_bias=True, name="z_gate_c", activation=tf.sigmoid)
				raw_dec_states = tf.concat([c_gate * enc_state.c, h_gate * enc_state.h, z], 1)
				#raw_dec_states = LSTMStateTuple(tf.concat([c_gate * enc_state.c, z], 1), tf.concat([h_gate * enc_state.h, z], 1))
			else:
				gate = tf.layers.dense(z, int(enc_state.get_shape()[1]), use_bias=True, name="z_gate", activation=tf.sigmoid)
				raw_dec_states = tf.concat([gate * enc_state, z], 1)

		# add BOW loss
		#num_hidden_units = int(math.sqrt(conf.output_vocab_size * int(decision_state.shape[1])))
		#bow_l1 = layers_core.Dense(num_hidden_units, use_bias=True, name="bow_hidden", activation=tf.tanh)
		#bow_l2 = layers_core.Dense(conf.output_vocab_size, use_bias=True, name="bow_out", activation=None)
		#bow = bow_l2(bow_l1(decision_state)) 

		#y_dec_inps = tf.slice(self.dec_inps, [0, 1], [-1, -1])
		#bow_y = tf.reduce_sum(tf.one_hot(y_dec_inps, on_value=1.0, off_value=0.0, axis=-1, depth=conf.output_vocab_size), axis=1)
		#batch_bow_losses = tf.reduce_sum(bow_y * (-1.0) * tf.nn.log_softmax(bow), axis=1)

		max_mem_size = conf.input_max_len + conf.output_max_len + 2
		with tf.name_scope("ShapeToBeam"):
			beam_raw_dec_states = nest.map_structure(lambda x:tile_batch(x, beam_size), raw_dec_states)
			beam_memory = nest.map_structure(lambda x:tile_batch(x, beam_size), enc_outs)
			beam_memory_lens = tf.squeeze(nest.map_structure(lambda x:tile_batch(x, beam_size), tf.expand_dims(inputs["enc_lens:0"], 1)), 1)
			beam_z = nest.map_structure(lambda x:tile_batch(x, beam_size), z)

		#def _to_beam(t):
		#	beam_t = tf.reshape(tf.tile(t, [1, beam_size]), [-1, int(t.get_shape()[1])])
		#	return beam_t 
		#with tf.name_scope("ShapeToBeam") as scope: 
		#	beam_raw_dec_states = tf.contrib.framework.nest.map_structure(_to_beam, raw_dec_states) 
		#	beam_memory = tf.reshape(tf.tile(self.enc_outs, [1, 1, beam_size]), [-1, conf.input_max_len, mem_size])
		#	beam_memory_lens = tf.squeeze(tf.reshape(tf.tile(tf.expand_dims(inputs["enc_lens:0"], 1), [1, beam_size]), [-1, 1]), 1)
		#	beam_z = tf.contrib.framework.nest.map_structure(_to_beam, z)
			
		#cell = AttnCell(cell_model=conf.cell_model, num_units=mem_size, num_layers=conf.num_layers,
		#				attn_type=conf.attention, memory=beam_memory, mem_lens=beam_memory_lens,
		#				max_mem_size=max_mem_size, addmem=conf.addmem, z=beam_z, keep_prob=conf.keep_prob,
		#				dtype=tf.float32)
		#with tf.variable_scope("DynDecode/AttnCell") as dyn_scope:
		decoder_multi_rnn_cells = CreateMultiRNNCell(conf.cell_model, num_units=mem_size, num_layers=conf.num_layers, output_keep_prob=conf.keep_prob)
		zero_cell_states = DecCellStateInit(beam_raw_dec_states, decoder_multi_rnn_cells, name="InitCell")

		attn_cell = AttnCellWrapper(cell=decoder_multi_rnn_cells, cell_init_states=zero_cell_states, attn_type=conf.attention,
									attn_size=mem_size, memory=beam_memory, mem_lens=beam_memory_lens, max_mem_size=max_mem_size,
									addmem=conf.addmem, z=beam_z, dtype=tf.float32, name="AttnWrapper")
			
		dec_init_state = None if self.conf.attention else zero_cell_states
		with tf.variable_scope("OutProj"):
			graphlg.info("Creating out_proj...") 
			if conf.out_layer_size:
				w = tf.get_variable("proj_w", [conf.out_layer_size, conf.output_vocab_size], dtype=dtype)
			else:
				w = tf.get_variable("proj_w", [mem_size, conf.output_vocab_size], dtype=dtype)
			b = tf.get_variable("proj_b", [conf.output_vocab_size], dtype=dtype)
			out_proj = (w, b)

		if not for_deploy: 
			hp_train = helper1_2.ScheduledEmbeddingTrainingHelper(inputs=emb_dec_next_inps, sequence_length=inputs["dec_lens:0"], embedding=self.embedding,
																sampling_probability=0.0, out_proj=out_proj)
			output_layer = layers_core.Dense(conf.out_layer_size, use_bias=True) if conf.out_layer_size else None
			my_decoder = basic_decoder1_2.BasicDecoder(cell=attn_cell, helper=hp_train, initial_state=dec_init_state, output_layer=output_layer)
			cell_outs, final_state, seq_len = decoder1_2.dynamic_decode(decoder=my_decoder, impute_finished=True, maximum_iterations=conf.output_max_len + 1)

			#cell_outs = tf.Print(cell_outs, [tf.shape(cell_outs)], message="cell_outs_shape")
			with tf.name_scope("Logits"):
				L = tf.shape(cell_outs.rnn_output)[1]
				rnn_output = tf.reshape(cell_outs.rnn_output, [-1, int(out_proj[0].shape[0])])
				rnn_output = tf.matmul(rnn_output, out_proj[0]) + out_proj[1] 
				logits = tf.reshape(rnn_output, [-1, L, int(out_proj[0].shape[1])])

			with tf.name_scope("DebugOutputs") as scope:
				outputs = tf.argmax(logits, axis=2)
				outputs = tf.reshape(outputs, [-1, L])
				outputs = self.out_table.lookup(tf.cast(outputs, tf.int64))

			# branch 2 for loss
			with tf.name_scope("Loss") as scope:
				tars = tf.slice(dec_inps, [0, 1], [-1, L])
				# wgts may be a more complicated form, for example a partial down-weighting of a sequence
				# but here i just use  1.0 weights for all no-padding label
				wgts = tf.cumsum(tf.one_hot(inputs["dec_lens:0"], L), axis=1, reverse=True)
				#wgts = wgts * tf.expand_dims(self.down_wgts, 1)
				loss_matrix = loss.sequence_loss(logits=logits, targets=tars, weights=wgts, average_across_timesteps=False, average_across_batch=False)
				#bow_loss = tf.reduce_sum(batch_bow_losses * self.down_wgts) / batch_wgt
				example_total_wgts = tf.reduce_sum(wgts, 1)
				total_wgts = tf.reduce_sum(example_total_wgts) 

				example_losses = tf.reduce_sum(loss_matrix, 1)
				see_loss = tf.reduce_sum(example_losses) / total_wgts

				KLD = tf.reduce_sum(KLD * example_total_wgts) / total_wgts 
				self.loss = tf.reduce_sum(example_losses + conf.kld_ratio * KLD) / total_wgts 

			with tf.name_scope(self.model_kind):
				tf.summary.scalar("loss", see_loss)
				tf.summary.scalar("kld", KLD) 
				#tf.summary.scalar("bow", bow_loss)
				for each in tf.trainable_variables():
					tf.summary.histogram(each.name, each)

			graph_nodes = {
				"loss":self.loss,
				"inputs":inputs,
				"debug_outputs":outputs,
				"outputs":{},
				"visualize":None
			}
			return graph_nodes
		else:
			beam_batch_size = tf.shape(beam_memory_lens)[0]
			hp_infer = helper1_2.GreedyEmbeddingHelper(embedding=self.embedding, start_tokens=tf.ones([beam_batch_size], dtype=tf.int32),
														end_token=EOS_ID, out_proj=out_proj)
			output_layer = layers_core.Dense(conf.out_layer_size, use_bias=True) if conf.out_layer_size else None
			my_decoder = beam_decoder.BeamDecoder(cell=attn_cell, helper=hp_infer, out_proj=out_proj, initial_state=dec_init_state, beam_splits=conf.beam_splits,
													max_res_num=conf.max_res_num, output_layer=output_layer)
			#cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, scope=scope, maximum_iterations=conf.output_max_len)
			cell_outs, final_state, seq_len = decoder1_2.dynamic_decode(decoder=my_decoder, impute_finished=True, maximum_iterations=conf.output_max_len + 1)

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
			batch_size = beam_batch_size / beam_size
			batch_size = tf.Print(batch_size, [batch_size], message="BATCH")

			#beam_symbols = tf.Print(cell_outs.beam_symbols, [tf.shape(cell_outs.beam_symbols)], message="beam_symbols")
			#beam_parents = tf.Print(cell_outs.beam_parents, [tf.shape(cell_outs.beam_parents)], message="beam_parents")
			#beam_ends = tf.Print(cell_outs.beam_ends, [tf.shape(cell_outs.beam_ends)], message="beam_ends") 
			#beam_end_parents = tf.Print(cell_outs.beam_end_parents, [tf.shape(cell_outs.beam_end_parents)], message="beam_end_parents") 
			#beam_end_probs = tf.Print(cell_outs.beam_end_probs, [tf.shape(cell_outs.beam_end_probs)], message="beam_end_probs") 
			#alignments = tf.Print(cell_outs.alignments, [tf.shape(cell_outs.alignments)], message="beam_attns")

			batch_offset = tf.expand_dims(tf.cumsum(tf.ones([batch_size, beam_size], dtype=tf.int32) * beam_size, axis=0, exclusive=True), 2)
			offset2 = tf.expand_dims(tf.cumsum(tf.ones([batch_size, beam_size * 2], dtype=tf.int32) * beam_size, axis=0, exclusive=True), 2)

			out_len = tf.shape(beam_symbols)[1]
			self.beam_symbol_strs = tf.reshape(self.out_table.lookup(tf.cast(beam_symbols, tf.int64)), [batch_size, beam_size, -1])
			self.beam_parents = tf.reshape(beam_parents, [batch_size, beam_size, -1]) - batch_offset

			self.beam_ends = tf.reshape(beam_ends, [batch_size, beam_size * 2, -1])
			self.beam_end_parents = tf.reshape(beam_end_parents, [batch_size, beam_size * 2, -1]) - offset2
			self.beam_end_probs = tf.reshape(beam_end_probs, [batch_size, beam_size * 2, -1])
			self.beam_attns = tf.reshape(alignments, [batch_size, beam_size, out_len, -1])

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
				"visualize":{"z":z}
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
		var_map = var_list
		#var_map = {}
		#for each in var_list:
		#	name = each.name
		#	#name = re.sub("lstm_cell/bias", "lstm_cell/biases", name)
		#	#name = re.sub("lstm_cell/kernel", "lstm_cell/weights", name)
		#	#name = re.sub("gates/bias", "gates/biases", name)
		#	#name = re.sub("candidate/bias", "candidate/biases", name)
		#	#name = re.sub("gates/kernel", "gates/weights", name)
		#	#name = re.sub("candidate/kernel", "candidate/weights", name)
		#	#name = re.sub("bias", "biases", name)
		#	#name = re.sub("dense/weights", "dense/kernel", name)
		#	#name = re.sub("dense/biases", "dense/bias", name)
		#	name = re.sub(":0", "", name)
		#	var_map[name] = each

		restorer = tf.train.Saver(var_list=var_map)
		return restorer

	def after_proc(self, out):
		outputs, probs, attns = Nick_plan.handle_beam_out(out, self.conf.beam_splits)

		outs = [[(outputs[n][i], probs[n][i]) for i in range(len(outputs[n]))] for n in range(len(outputs))]

		#sorted_outs = sorted(outs, key=lambda x:x[1]/len(x[0]), reverse=True)
		sorted_outs = [sorted(outs[n], key=lambda x:x[1], reverse=True) for n in range(len(outs))]
		after_proc_out = [[{"outputs":res[0], "probs":res[1], "model_name":self.name} for res in example] for example in sorted_outs]
		return after_proc_out 
