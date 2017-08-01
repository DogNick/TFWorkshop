import sys
import time
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.contrib import lookup

from tensorflow.contrib.rnn import  MultiRNNCell, AttentionCellWrapper, GRUCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn

from tensorflow.contrib.seq2seq.python.ops import loss
from tensorflow.python.layers import core as layers_core
from ModelCore import *


def normalize(inputs, 
			  epsilon = 1e-8,
			  scope="ln",
			  reuse=None):
	'''Applies layer normalization.
	
	Args:
	  inputs: A tensor with 2 or more dimensions, where the first dimension has
		`batch_size`.
	  epsilon: A floating number. A very small number for preventing ZeroDivision Error.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
	  
	Returns:
	  A tensor with the same shape and data dtype as `inputs`.
	'''
	with tf.variable_scope(scope, reuse=reuse):
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]
	
		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		#beta= tf.Variable(tf.zeros(params_shape))
		beta= tf.get_variable(name="beta", shape=params_shape, initializer=tf.zeros_initializer())
		#gamma = tf.Variable(tf.ones(params_shape))
		gamma = tf.get_variable(name="gamma", shape=params_shape, initializer=tf.ones_initializer())
		normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
		outputs = gamma * normalized + beta
		
	return outputs

def embedding(inputs, 
			  vocab_size, 
			  num_units, 
			  zero_pad=True, 
			  scale=True,
			  scope="embedding", 
			  reuse=None):
	'''Embeds a given tensor.

	Args:
	  inputs: A `Tensor` with type `int32` or `int64` containing the ids
		 to be looked up in `lookup table`.
	  vocab_size: An int. Vocabulary size.
	  num_units: An int. Number of embedding hidden units.
	  zero_pad: A boolean. If True, all the values of the fist row (id 0)
		should be constant zeros.
	  scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.

	Returns:
	  A `Tensor` with one more rank than inputs's. The last dimensionality
		should be `num_units`.
		
	For example,
	
	```
	import tensorflow as tf
	
	inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
	outputs = embedding(inputs, 6, 2, zero_pad=True)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(outputs)
	>>
	[[[ 0.		  0.		]
	  [ 0.09754146  0.67385566]
	  [ 0.37864095 -0.35689294]]

	 [[-1.01329422 -1.09939694]
	  [ 0.7521342   0.38203377]
	  [-0.04973143 -0.06210355]]]
	```
	
	```
	import tensorflow as tf
	
	inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
	outputs = embedding(inputs, 6, 2, zero_pad=False)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(outputs)
	>>
	[[[-0.19172323 -0.39159766]
	  [-0.43212751 -0.66207761]
	  [ 1.03452027 -0.26704335]]

	 [[-0.11634696 -0.35983452]
	  [ 0.50208133  0.53509563]
	  [ 1.22204471 -0.96587461]]]	
	```	
	'''
	with tf.variable_scope(scope, reuse=reuse):
		lookup_table = tf.get_variable('lookup_table',
									   dtype=tf.float32,
									   shape=[vocab_size, num_units],
									   initializer=tf.layers.xavier_initializer())
		if zero_pad:
			lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
									  lookup_table[1:, :]), 0)
		outputs = tf.nn.embedding_lookup(lookup_table, inputs)
		
		if scale:
			outputs = outputs * (num_units ** 0.5) 
			
	return outputs
	
def multihead_attention(queries, 
						keys, 
						num_units=None, 
						num_heads=8, 
						dropout_rate=0,
						use_dropout=True,
						causality=False,
						scope="multihead_attention", 
						reuse=None):
	'''Applies multihead attention.
	
	Args:
	  queries: A 3d tensor with shape of [N, T_q, C_q].
	  keys: A 3d tensor with shape of [N, T_k, C_k].
	  num_units: A scalar. Attention size.
	  dropout_rate: A floating point number.
	  use_dropout: Boolean. Controller of mechanism for dropout.
	  causality: Boolean. If true, units that reference the future are masked. 
	  num_heads: An int. Number of heads.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
		
	Returns
	  A 3d tensor with shape of (N, T_q, C)  
	'''
	with tf.variable_scope(scope, reuse=reuse):
		# Set the fall back option for num_units
		if num_units is None:
			num_units = queries.get_shape().as_list[-1]
		
		# Linear projections
		#queries = tf.Print(queries, [tf.shape(queries)], message="queries")
		Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
		K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
		V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
		
		#Q = tf.Print(Q, [tf.shape(Q)], message="QQQQQQQQQQQQQQQQQQ")
		# Split and concat
		Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
		K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
		V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
		#Q_ = tf.Print(Q_, [tf.shape(Q_)], message="concated QQQQQQQQQQQQQQQQQQ")

		# Multiplication
		outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
		
		# Scale
		outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
		
		# Key Masking
		key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
		key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
		key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
		
		paddings = tf.ones_like(outputs)*(-2**32+1)
		outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
		# Causality = Future blinding
		if causality:
			diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
			tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
			masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
			paddings = tf.ones_like(masks)*(-2**32+1)
			outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
		# Activation
		outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
		 
		# Query Masking
		query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
		query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
		query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
		outputs *= query_masks # broadcasting. (N, T_q, C)
		  
		# Dropouts
		outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(use_dropout))
			   
		# Weighted sum
		outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
		
		# Restore shape
		outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)
			  
		# Residual connection
		outputs += queries
			  
		# Normalize
		outputs = normalize(outputs) # (N, T_q, C)
 
	return outputs

def feedforward(inputs, 
				num_units=[2048, 512],
				scope="feedforward", 
				reuse=None):
	'''Point-wise feed forward net.
	
	Args:
	  inputs: A 3d tensor with shape of [N, T, C].
	  num_units: A list of two integers.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
		
	Returns:
	  A 3d tensor with the same shape and dtype as inputs
	'''
	with tf.variable_scope(scope, reuse=reuse):
		# Inner layer
		params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
				  "activation": tf.nn.relu, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		
		# Readout layer
		params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
				  "activation": None, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		
		# Residual connection
		outputs += inputs
		
		# Normalize
		outputs = normalize(outputs)
	
	return outputs

def label_smoothing(inputs, epsilon=0.1):
	'''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
	
	Args:
	  inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
	  epsilon: Smoothing rate.
	
	For example,
	
	```
	import tensorflow as tf
	inputs = tf.convert_to_tensor([[[0, 0, 1], 
	   [0, 1, 0],
	   [1, 0, 0]],

	  [[1, 0, 0],
	   [1, 0, 0],
	   [0, 1, 0]]], tf.float32)
	   
	outputs = label_smoothing(inputs)
	
	with tf.Session() as sess:
		print(sess.run([outputs]))
	
	>>
	[array([[[ 0.03333334,  0.03333334,  0.93333334],
		[ 0.03333334,  0.93333334,  0.03333334],
		[ 0.93333334,  0.03333334,  0.03333334]],

	   [[ 0.93333334,  0.03333334,  0.03333334],
		[ 0.93333334,  0.03333334,  0.03333334],
		[ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
	```	
	'''
	K = inputs.get_shape().as_list()[-1] # number of channels
	return ((1-epsilon) * inputs) + (epsilon / K)


def AttnEncode(enc, num_heads, for_deploy, num_blocks, hidden_units, dropout_rate):
	## Blocks
	for i in range(num_blocks):
		with tf.variable_scope("num_blocks_{}".format(i)):
			### Multihead Attention
			enc_attn = multihead_attention(queries=enc, 
									  keys=enc, 
									  num_units=hidden_units, 
									  num_heads=num_heads, 
									  dropout_rate=dropout_rate,
									  use_dropout=for_deploy,
									  causality=False)
			### Feed Forward
			enc = feedforward(enc_attn, num_units=[4 * hidden_units, hidden_units])
	return enc



def AttnDecode(enc_mem, dec, num_heads, for_deploy, num_blocks, hidden_units, dropout_rate):
	## Blocks
	for i in range(num_blocks):
		with tf.variable_scope("num_blocks_{}".format(i)):
			## Multihead Attention ( self-attention)
			#dec_emb = tf.Print(dec_emb, [tf.shape(dec_emb)], message="dec_emb")
			dec_self_attn = multihead_attention(queries=dec, 
											keys=dec, 
											num_units=hidden_units, 
											num_heads=num_heads, 
											dropout_rate=dropout_rate,
											use_dropout=for_deploy,
											causality=True, 
											scope="self_attention")
						
			## Multihead Attention ( vanilla attention)
			#dec_self_attn = tf.Print(dec_self_attn, [tf.shape(dec_self_attn)], message="dec_self_attn")
			multi_head_attn_on_encs = multihead_attention(queries=dec_self_attn, 
											keys=enc_mem, 
											num_units=hidden_units, 
											num_heads=num_heads,
											dropout_rate=dropout_rate,
											use_dropout=for_deploy,
											causality=False,
											scope="vanilla_attention")
			## Feed Forward
			dec = feedforward(multi_head_attn_on_encs, num_units=[4 * hidden_units, hidden_units])

	return dec 

class AllAttn(ModelCore):
	def __init__(self, name, job_type="single", task_id=0, dtype=tf.float32):
		super(AllAttn, self).__init__(name, job_type, task_id, dtype) 
		self.embedding = None
		self.out_proj = None

	def build(self, for_deploy, variants=""):
		conf = self.conf
		name = self.name
		job_type = self.job_type
		dtype = self.dtype

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

		self.enc_str_inps = tf.placeholder(tf.string, shape=[None, conf.input_max_len], name="enc_inps") 
		self.dec_str_inps = tf.placeholder(tf.string, shape=[None, conf.output_max_len + 1], name="dec_inps") 

		# lookup
		self.enc_inps = self.in_table.lookup(self.enc_str_inps)
		self.dec_tar_inps = self.in_table.lookup(self.dec_str_inps)

		batch_size = tf.shape(self.enc_inps)[0]

		with tf.variable_scope("encoder"):
			## Encoding embedding
			with tf.device("/cpu:0"):
				self.embedding = tf.get_variable('embedding', dtype=tf.float32,
										   shape=[self.conf.input_vocab_size, self.conf.embedding_size],
										   initializer=tf.contrib.layers.xavier_initializer())
				self.embedding = tf.concat((tf.zeros(shape=[1, self.conf.embedding_size]), self.embedding[1:, :]), 0)
				self.pos_embedding = tf.get_variable('positional_embedding', dtype=tf.float32,
										   shape=[self.conf.input_max_len, self.conf.embedding_size],
										   initializer=tf.contrib.layers.xavier_initializer())
				# Word embedding
				self.enc = tf.nn.embedding_lookup(self.embedding, self.enc_inps)
				## Positional Encoding
				pos = tf.tile(tf.expand_dims(tf.range(tf.shape(self.enc_inps)[1]), 0), [tf.shape(self.enc_inps)[0], 1])
				pos_enc = tf.nn.embedding_lookup(self.pos_embedding, pos)

			self.enc = self.enc * (self.conf.embedding_size ** 0.5) + pos_enc
			# dropout
			self.enc = tf.layers.dropout(self.enc, rate=self.conf.dropout_rate, training=tf.convert_to_tensor(not for_deploy))
			# Attn Blocks
			self.enc = AttnEncode(self.enc, self.conf.num_heads, for_deploy, self.conf.num_blocks, self.conf.hidden_units, self.conf.dropout_rate)

		with tf.variable_scope("decoder"):
			## Decoding embedding 
			with tf.device("/cpu:0"):
				self.dec_embedding = tf.get_variable('dec_embedding', dtype=tf.float32,
										   shape=[self.conf.output_vocab_size, self.conf.embedding_size],
										   initializer=tf.contrib.layers.xavier_initializer())
				self.dec_embedding = tf.concat((tf.zeros(shape=[1, self.conf.embedding_size]), self.embedding[1:, :]), 0)
				self.pos_dec_embedding = tf.get_variable('positional_embedding', dtype=tf.float32,
										   shape=[self.conf.output_max_len + 1, self.conf.embedding_size],
										   initializer=tf.contrib.layers.xavier_initializer())
			if for_deploy:
				#inps = tf.ones([batch_size, 1], tf.int32)
				dec_inps = tf.zeros([batch_size, conf.output_max_len + 1], tf.int32)
				time = constant_op.constant(0, tf.int32) 
			else:
				dec_inps = tf.to_int32(self.dec_tar_inps)
				#inps = tf.Print(inps, [tf.shape(inps)], message="inps", summarize=10000)
				time = constant_op.constant(conf.output_max_len, tf.int32)

			def condition(time, finished, inps, logits, keys):
				finished = tf.equal(time, tf.convert_to_tensor(self.conf.output_max_len + 1))
				return tf.logical_not(finished)
									
			def autoregressive(time, finished, inps, logits, keys):
				# inps are self.conf.output_max_len with one EOS at tail
				# here first remove the last EOS and head it by GO (1) the output should be 
				# the original input sequence at training time, during which the loop body run only once
				inps = tf.concat([tf.ones([batch_size, 1], tf.int32), tf.slice(inps, [0, 0], [-1, self.conf.output_max_len])], 1)
				
				with tf.device("/cpu:0"):
					dec_emb = tf.nn.embedding_lookup(self.dec_embedding, inps)

				pos = tf.tile(tf.expand_dims(tf.range(tf.shape(inps)[1]), 0), [tf.shape(inps)[0], 1])

				with tf.device("/cpu:0"):
					pos_dec_emb = tf.nn.embedding_lookup(self.pos_dec_embedding, pos)

				dec_emb = dec_emb * (self.conf.embedding_size ** 0.5) + pos_dec_emb
				dec_emb = tf.layers.dropout(dec_emb, rate=self.conf.dropout_rate, training=tf.convert_to_tensor(not for_deploy))
				dec_out = AttnDecode(keys, dec_emb, self.conf.num_heads, for_deploy, self.conf.num_blocks, self.conf.hidden_units, self.conf.dropout_rate)

				## Dropout
				# Final linear projection
				logits = tf.layers.dense(dec_out, self.conf.output_vocab_size)
				preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
				time = time + 1
				next_inps = preds
				#time = tf.Print(time, [time], message="t", summarize=1000)
				#preds = tf.Print(preds, [preds], message="preds", summarize=10000)
				return time, finished, next_inps, logits, keys

			finished = constant_op.constant(True, tf.bool)
			logits = tf.zeros([batch_size, self.conf.output_max_len + 1, self.conf.output_vocab_size]) 

			(time, finished, preds, 
				logits, keys) = tf.while_loop(
								    condition,
								    autoregressive,
								    loop_vars=[
								    	 time,
								    	 finished,
								    	 dec_inps,
								    	 logits,
								    	 self.enc
								    ],
								    shape_invariants=[
								        time.get_shape(),
								        finished.get_shape(),
								        tf.TensorShape([None, None]),
								        tf.TensorShape([None, None, None]),
								        self.enc.get_shape()
								    ]
							    )
			self.preds = preds
			self.logits = logits

			#self.preds = tf.Print(self.preds, [tf.shape(self.preds)], message="preds")
			#self.logits = tf.Print(self.logits, [tf.shape(self.logits)], message="logits")

		if not for_deploy:  
			self.istarget = tf.to_float(tf.not_equal(self.dec_tar_inps, 0))
			self.acc = tf.reduce_sum(tf.to_float(tf.equal(tf.to_int64(self.preds), self.dec_tar_inps)) * self.istarget) / (tf.reduce_sum(self.istarget))
			self.pred_strs = self.out_table.lookup(tf.cast(self.preds, tf.int64))

			# Loss
			# smoothing
			#self.y_smoothed = label_smoothing(tf.one_hot(self.dec_tar_inps, depth=self.conf.output_vocab_size))
			self.y_smoothed = tf.one_hot(self.dec_tar_inps, depth=self.conf.output_vocab_size, axis=-1)
			self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
			self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

			# Summary 
			tf.summary.scalar('mean_loss', self.mean_loss)
			tf.summary.scalar('acc', self.acc)
			outputs = self.pred_strs
			return self.mean_loss, {}, outputs  
		else:
			self.pred_strs = self.out_table.lookup(tf.cast(self.preds, tf.int64))
			inputs = {"enc_inps":self.enc_str_inps}
			outputs = self.pred_strs
			return None, inputs, outputs  

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
		var_list = self.global_params + self.trainable_params + self.optimizer_params + \
					tf.get_default_graph().get_collection("saveable_objects")
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
			##name = re.sub("bias", "biases", name)
			##name = re.sub("dense/weights", "dense/kernel", name)
			##name = re.sub("dense/biases", "dense/bias", name)
			name = re.sub(":0", "", name)
			var_map[name] = each
		restorer = tf.train.Saver(var_list=var_map)
		return restorer

	def preproc(self, records, use_seg=True, for_deploy=False, default_wgt=1.0):
		data = []
		for each in records:
			if for_deploy:
				p = each.strip()
				words, _ = tokenize_word(p) if use_seg else (p.split(), None)
				p_list = words #re.split(" +", p.strip())
				r_list = []
				data.append([p_list, r_list])
			else:
				segs = re.split("\t", each.strip())
				if len(segs) < 2:
					continue
				p, r = segs[0], segs[1]
				p_list = re.split(" +", p)
				r_list = re.split(" +", r)
				down_wgts = segs[-1] if len(segs) > 2 else default_wgt 
				if self.conf.reverse:
					p_list, r_list = r_list, p_list
				data.append([p_list, r_list])

		conf = self.conf
		batch_enc_inps, batch_dec_inps = [], []
		for encs, decs in data:
			encs = encs[0:conf.input_max_len]
			enc_inps = encs + ["_PAD"] * (conf.input_max_len - len(encs))
			batch_enc_inps.append(enc_inps)
			if not for_deploy:
				decs += ["_EOS"]
				decs = decs[0:conf.output_max_len + 1]
				batch_dec_inps.append(decs + ["_PAD"] * (conf.output_max_len + 1 - len(decs)))

		feed_dict = {
			"enc_inps:0": batch_enc_inps,
			"dec_inps:0": batch_dec_inps,
		}
		for k, v in feed_dict.items():
			if not v: 
				del feed_dict[k]
		return feed_dict

	def after_proc(self, out):
		after_proc_out = {
			"outputs":out["outputs"],
		}
		return after_proc_out

	def print_after_proc(self, after_proc):
		for each in after_proc["outputs"]:
			out_str = "".join(each)
			print out_str[0:out_str.find("_EOS")]
	

if __name__ == "__main__":
	name = "allattn"
	model = AllAttn(name)
	#with tf.device("/cpu:0"):
	#	model.build(for_deploy=False)
	#init_ops = model.get_init_ops()
	#model.sess = tf.Session()
	#model.sess.run(init_ops)
	if len(sys.argv) == 2:
		gpu = 0
	flag = sys.argv[1]
	#model(flag, use_seg=False)
	model(flag, use_seg=True)
