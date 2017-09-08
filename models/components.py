import tensorflow as tf
from nick_tf import dynamic_attention_wrapper
from nick_tf import attention_wrapper1_2
from nick_tf import decoder1_2 
from nick_tf import basic_decoder1_2

from tensorflow.contrib.rnn import  MultiRNNCell, GRUCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.python.layers import core as layers_core

DynamicAttentionWrapper = dynamic_attention_wrapper.DynamicAttentionWrapper
DynamicAttentionWrapperState = dynamic_attention_wrapper.DynamicAttentionWrapperState 
Bahdanau = dynamic_attention_wrapper.BahdanauAttention
Luong = dynamic_attention_wrapper.LuongAttention

AttentionWrapper = attention_wrapper1_2.AttentionWrapper
AttentionWrapperState = attention_wrapper1_2.AttentionWrapperState
Bahdanau1_2 = attention_wrapper1_2.BahdanauAttention
Luong1_2 = attention_wrapper1_2.LuongAttention



import logging as log
graphlg = log.getLogger("graph")

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


def DynEncode(cell_model, num_units, num_layers, emb_inps, enc_lens, keep_prob=1.0, bidi=False, name_scope="encoder", dtype=tf.float32):
	if bidi:
		cell_fw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, name_scope="cell_fw")
		cell_bw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, name_scope="cell_bw")
		enc_outs, enc_states = bidirectional_dynamic_rnn(cell_fw=cell_fw,
														cell_bw=cell_bw,
														inputs=emb_inps,
														sequence_length=enc_lens,
														dtype=dtype,
														parallel_iterations=16,
														scope=name_scope)
		fw_s, bw_s = enc_states 
		enc_states = []
		with tf.name_scope(name_scope):
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
	
# Dynamic RNN creater for specific cell_model, num_units, num_layers, etc
def DynRNN(cell_model, num_units, num_layers, emb_inps, enc_lens, keep_prob=1.0, bidi=False, name_scope="encoder", dtype=tf.float32):
	"""A Dynamic RNN Creator"
		Take embedding inputs and make dynamic rnn process 
	"""
	with tf.name_scope(name_scope):
		if bidi:
			cell_fw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, name_scope="cell_fw")
			cell_bw = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, name_scope="cell_bw")
			enc_outs, enc_states = bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=emb_inps, sequence_length=enc_lens,
															dtype=dtype, parallel_iterations=16, scope=name_scope)
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
	"""Create a multi rnn cell object
		create multi layer cells with specific size, layers and drop prob
	"""
	#with tf.variable_scope(name_scope):
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

def AttnCellWrapper(cell, cell_init_states, attn_type, attn_size, memory, mem_lens, max_mem_size, addmem=False, z=None, normalize=False, dtype=tf.float32, name="AttnWrapper"):
	with tf.variable_scope(name) as scope:
		if attn_type == "Luo":
			mechanism = Luong1_2(num_units=attn_size, memory=memory, max_mem_size=max_mem_size, memory_sequence_length=mem_lens, scale=normalize, z=z, name=scope.original_name_scope)
		elif attn_type == "Bah":
			mechanism = Bahdanau1_2(num_units=attn_size, memory=memory, max_mem_size=max_mem_size, memory_sequence_length=mem_lens, scale=normalize, z=z, name=scope.original_name_scope)
		elif attn_type == None:
			mechanism = None
		else:
			print "Unknown attention stype, must be Luo or Bah" 
			exit(0)
		if mechanism == None: 
			return cell
		else:
			return AttentionWrapper(cell=cell, attention_mechanism=mechanism, attention_layer_size=attn_size, addmem=addmem, initial_cell_state=cell_init_states)

def AttnCell(cell_model, num_units, num_layers, memory, mem_lens, attn_type, max_mem_size, keep_prob=1.0, addmem=False, z=None, normalize=False, dtype=tf.float32, name_scope="AttnCell"):
	# Attention  
	"""Wrap a cell by specific attention mechanism with some memory
	Params:
		max_mem_size is for incremental enc memory (for addmem attention)
	"""
	print "Attention type %s" % str(attn_type)
	with tf.name_scope(name_scope):
		decoder_cell = CreateMultiRNNCell(cell_model, num_units, num_layers, keep_prob, False, name_scope)
		if attn_type == "Luo":
			mechanism = Luong(num_units=num_units, memory=memory, max_mem_size=max_mem_size, memory_sequence_length=mem_lens, normalize=normalize, z=z)
		elif attn_type == "Bah":
			mechanism = Bahdanau(num_units=num_units, memory=memory, max_mem_size=max_mem_size, memory_sequence_length=mem_lens, normalize=normalize, z=z)
		elif attn_type == None:
			return decoder_cell
		else:
			print "Unknown attention stype, must be Luo or Bah" 
			exit(0)
		attn_cell = DynamicAttentionWrapper(cell=decoder_cell, attention_mechanism=mechanism, attention_size=num_units, addmem=addmem)
		return attn_cell

def DecCellStateInit(all_enc_state, decoder_multi_rnn_cells, init_type="last2first", name="DecCellStateInit"):
	"""Create a init cell state for decoder(or attn wrapper) from all encoder states
	   this will create some different type of init cell state based on decoder cell type
	"""
	with tf.variable_scope(name):
		batch_size = tf.shape(all_enc_state)[0] 
		zero_states = decoder_multi_rnn_cells.zero_state(batch_size, tf.float32)
		init_states = []
		if init_type == "last2first":
			for i, each in enumerate(zero_states):
				if i >= 1:	
					init_states.append(each)
					continue
				if isinstance(each, LSTMStateTuple):
					init_h = tf.layers.dense(all_enc_state, each.h.get_shape()[1], name="proj_enclast_to_h")
					init_c = tf.layers.dense(all_enc_state, each.c.get_shape()[1], name="proj_enclast_to_c")
					init_states.append(LSTMStateTuple(init_c, init_h))
				else:
					init = tf.layers.dense(all_enc_state, each.get_shape()[1], name="proj_enclast")
					init_states.append(init)
			return tuple(init_states)
		elif init_type == "allzeros":
			return None
		else:	
			print "init type %s unknonw !!!" % init_type
			exit(0)

def DecStateInit(all_enc_states, decoder_cell, batch_size, init_type="each2each", use_proj=True, name="DecStateInit"):
	"""make init states for decoder cells
		take some states (maybe for each encoder layers) to make different
		type of init states for decoder cells
	"""
	# Encoder states for initial state, with vae 
	with tf.name_scope(name):
		# get decoder zero_states as a default and shape guide
		zero_states = decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
		if isinstance(zero_states, DynamicAttentionWrapperState) or isinstance(zero_states, AttentionWrapperState):
			dec_zero_states = zero_states.cell_state
		else:
			dec_zero_states = zero_states
		
		#TODO check all_enc_states

		init_states = []
		if init_type == "each2each":
			for i, each in enumerate(dec_zero_states):
				if i >= len(all_enc_states):	
					init_states.append(each)
					continue
				if use_proj == False:
					init_states.append(all_enc_states[i])
					continue
				enc_state = all_enc_states[i]
				if isinstance(each, LSTMStateTuple):
					init_h = tf.layers.dense(enc_state.h, each.h.get_shape()[1], name="proj_l%d_to_h" % i)
					init_c = tf.layers.dense(enc_state.c, each.c.get_shape()[1], name="proj_l%d_to_c" % i)
					init_states.append(LSTMStateTuple(init_c, init_h))
				else:
					init = tf.layers.dense(enc_state, each.get_shape()[1], name="ToDecShape")
					init_states.append(init)
		elif init_type == "all2first":
			enc_state = tf.concat(all_enc_states, 1)
			dec_state = dec_zero_states[0]
			if isinstance(dec_state, LSTMStateTuple):
				init_h = tf.layers.dense(enc_state, dec_state.h.get_shape()[1], name="ToDecShape_h")
				init_c = tf.layers.dense(enc_state, dec_state.c.get_shape()[1], name="ToDecShape_c")
				init_states.append(LSTMStateTuple(init_c, init_h))
			else:
				init = tf.layers.dense(enc_state, dec_state.get_shape()[1], name="ToDecShape")
				init_states.append(init)
			init_states.extend(dec_zero_states[1:])
		elif init_type == "allzeros":
			init_states = dec_zero_states
		else:	
			print "init type %s unknonw !!!" % init_type
			exit(0)

		if isinstance(decoder_cell, DynamicAttentionWrapper):
			zero_states = DynamicAttentionWrapperState(tuple(init_states), zero_states.attention, zero_states.newmem, zero_states.alignments)
		elif isinstance(decoder_cell, AttentionWrapper):
			zero_states = AttentionWrapperState(tuple(init_states), zero_states.attention, zero_states.time, zero_states.alignments, zero_states.alignment_history, zero_states.newmem)
		else:
			zero_states = tuple(init_states)
		
		return zero_states

def Ptheta(states, hidden_units, enc_latent_dim, stddev=1.0, prior_type="mlp", name_scope=None):
	all_states = []
	for each in states:
		all_states.extend(list(each))
	state = tf.concat(all_states, 1, name="concat_states")
	with tf.name_scope(name_scope, "Ptheta"):
		epsilon = tf.random_normal([tf.shape(state)[0], enc_latent_dim], name="epsilon", stddev=stddev)
		if prior_type == "mlp":
			mu_layer1 = layers_core.Dense(hidden_units, use_bias=True, name="mu_layer1", activation=None)
			mu_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="mu_layer2")
			logvar_layer1 = layers_core.Dense(hidden_units, use_bias=True, name="logvar_layer1", activation=None)
			logvar_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="logvar_layer2")
			mu_prior = mu_layer2(relu(mu_layer1(state)))
			logvar_prior = logvar_layer2(relu(logvar_layer1(state)))
			z = mu_prior + tf.exp(0.5 * logvar_prior) * epsilon
		elif prior_type == "simple":
			mu_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="mu_layer2", activation=tf.tanh)
			logvar_layer2 = layers_core.Dense(enc_latent_dim, use_bias=True, name="logvar_layer2", activation=tf.tanh)
			mu_prior = mu_layer2(state)
			logvar_prior = logvar_layer2(state)
			z = mu_prior + tf.exp(0.5 * logvar_prior) * epsilon
		else:
			z = epsilon
			mu_prior = tf.zeros_like(epsilon)
			logvar_prior = 2 * tf.log(stddev) * tf.ones_like(epsilon)
		return z, mu_prior, logvar_prior 

def PriorNet(states, hidden_units, enc_latent_dim, stddev=1.0, prior_type="mlp"):
	all_states = []
	for each in states:
		all_states.extend(list(each))
	state = tf.concat(all_states, 1, name="concat_states")
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
		mu_prior = tf.zeros_like(epsilon)
		logvar_prior = 2 * tf.log(stddev) * tf.ones_like(epsilon)
	return z, mu_prior, logvar_prior 

def VAE(states, enc_latent_dim, mu_prior=None, logvar_prior=None, reuse=False, dtype=tf.float32, name_scope=None):
	#with tf.variable_scope(name_scope) as scope:
	graphlg.info("Creating latent z for encoder states") 
	all_states = []
	for each in states:
		all_states.extend(list(each))
	h_state = tf.concat(all_states, 1, name="concat_states")
	with tf.variable_scope("EncToLatent"):
		epsilon = tf.random_normal([tf.shape(h_state)[0], enc_latent_dim])
		W_enc_hidden_mu = tf.Variable(tf.random_normal([int(h_state.get_shape()[1]), enc_latent_dim]),name="w_enc_hidden_mu")
		b_enc_hidden_mu = tf.Variable(tf.random_normal([enc_latent_dim]), name="b_enc_hidden_mu") 
		W_enc_hidden_logvar = tf.Variable(tf.random_normal([int(h_state.get_shape()[1]), enc_latent_dim]), name="w_enc_hidden_logvar")
		b_enc_hidden_logvar = tf.Variable(tf.random_normal([enc_latent_dim]), name="b_enc_hidden_logvar") 
		# Should there be any non-linearty?
		# A normal sampler
		mu_enc = tf.tanh(tf.matmul(h_state, W_enc_hidden_mu) + b_enc_hidden_mu)
		logvar_enc = tf.matmul(h_state, W_enc_hidden_logvar) + b_enc_hidden_logvar
		z = mu_enc + tf.exp(0.5 * logvar_enc) * epsilon

	if mu_prior == None:
		mu_prior = tf.zeros_like(epsilon)
	if logvar_prior == None:
		logvar_prior = tf.zeros_like(epsilon)

	# Should this z be concatenated by original state ?
	with tf.variable_scope("KLD"):
		KLD = -0.5 * tf.reduce_sum(1 + logvar_enc - logvar_prior - (tf.pow(mu_enc - mu_prior, 2) + tf.exp(logvar_enc))/tf.exp(logvar_prior), axis = 1)
	return z, KLD, None 

def CreateVAE(states, enc_latent_dim, mu_prior=None, logvar_prior=None, reuse=False, dtype=tf.float32, name_scope=None):
	"""Create vae states and kld with specific distribution
		encode all input states into a random variable, and create a KLD
		between prior and latent isolated Gaussian distribution
	"""
	with tf.name_scope(name_scope) as scope:
		graphlg.info("Creating latent z for encoder states") 
		all_states = []
		for each in states:
			all_states.extend(list(each))
		h_state = tf.concat(all_states, 1, name="concat_states")
		with tf.name_scope("EncToLatent"):
			epsilon = tf.random_normal([tf.shape(h_state)[0], enc_latent_dim])
			W_enc_hidden_mu = tf.Variable(tf.random_normal([int(h_state.get_shape()[1]), enc_latent_dim]),name="w_enc_hidden_mu")
			b_enc_hidden_mu = tf.Variable(tf.random_normal([enc_latent_dim]), name="b_enc_hidden_mu") 
			W_enc_hidden_logvar = tf.Variable(tf.random_normal([int(h_state.get_shape()[1]), enc_latent_dim]), name="w_enc_hidden_logvar")
			b_enc_hidden_logvar = tf.Variable(tf.random_normal([enc_latent_dim]), name="b_enc_hidden_logvar") 
			# Should there be any non-linearty?
			# A normal sampler
			mu_enc = tf.tanh(tf.matmul(h_state, W_enc_hidden_mu) + b_enc_hidden_mu)
			logvar_enc = tf.matmul(h_state, W_enc_hidden_logvar) + b_enc_hidden_logvar
			z = mu_enc + tf.exp(0.5 * logvar_enc) * epsilon

		if mu_prior == None:
			mu_prior = tf.zeros_like(epsilon)
		if logvar_prior == None:
			logvar_prior = tf.zeros_like(epsilon)

		# Should this z be concatenated by original state ?
		with tf.name_scope("KLD"):
			KLD = -0.5 * tf.reduce_sum(1 + logvar_enc - logvar_prior - (tf.pow(mu_enc - mu_prior, 2) + tf.exp(logvar_enc))/tf.exp(logvar_prior), axis = 1)
	return z, KLD, None 
