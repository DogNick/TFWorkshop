import os
import time
import datetime
import pickle
import numpy as np
import tensorflow as tf
from sklearn import metrics
from word_seg.tencent_segment import *
#from gensim.models.keyedvectors import KeyedVectors
#from KimCNNLayer import KimCNNLayer
#from EmbeddingWordLayer import EmbeddingWordLayer

class KimCNN(object):

	def __init__(self, name, for_deploy=False, job_type="single", dtype=tf.float32, data_dequeue_op=None):
		self.text_len = 40
		self.pad_word = '<PADDINGWORD/>'
		self.input_x_name = 'input_x'
		self.output_prob_name = 'prob'
	
	def get_dataset(self, train_set, dev_set):
		self.train_set = train_set
		self.dev_set = dev_set

	def get_conf(self, word_list, max_length, filter_sizes, num_filters,
			embedding_size, num_classes, l2_reg_lambda=0.0, word2vec=None, work_dir='.'):
		self.word_list = word_list
		self.max_length = max_length
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.embedding_size = embedding_size
		self.num_classes = num_classes
		self.l2_reg_lambda = l2_reg_lambda
		self.word2vec = word2vec
		self.work_dir = work_dir

		# default conf
		self.voc_file_path = './word_data/logQAWeibo/log.qa.weibo.corpus.wordlist'

		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		self.out_dir = self.work_dir + "/runs/" + timestamp
		#print("Writing to {}\n".format(self.out_dir))

	def build(self, for_deploy=False):
		self.w2id_table = tf.contrib.lookup.MutableHashTable(
				key_dtype=tf.string, value_dtype=tf.int64, default_value=0,
				shared_name="w2id_table", name="w2id_table", checkpoint=True)
		self.input_x = tf.placeholder(tf.string, [None, self.max_length], name="intput_x")
		self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
		if for_deploy:
			self.dropout_keep_prob = tf.constant(1.0)
		else:
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		layers = []
		input_x_id = self.w2id_table.lookup(self.input_x)
		layers.append(self.EmbeddingWordLayer(input_x_id, self.word_list.w2id, 
					self.word_list.size, self.embedding_size,word_vec_file=self.word2vec))
		layers.append(self.KimCNNLayer(layers[-1].output, self.max_length, 
					self.embedding_size,self.filter_sizes, self.num_filters, self.dropout_keep_prob))
		output = tf.contrib.layers.fully_connected(layers[-1].output, self.num_classes, activation_fn=None)
		logits = output

		l2_loss = 0
		for v in tf.trainable_variables():
			l2_loss += tf.nn.l2_loss(v)

		# Final (unnormalized) scores and predictions
		with tf.name_scope("loss"):
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits)
			self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			self.predictions = tf.argmax(logits, 1, name="predictions")
			self.probability = tf.nn.softmax(logits, name="probability")

		# nodes for export
		self.nodes = {
			"inputs":{self.input_x_name:self.input_x},
			"outputs":{self.output_prob_name:self.probability}
		}

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
			self.correct = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))

		self._lr = tf.Variable(0.0, trainable=False)
		self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.optimizer = tf.train.AdamOptimizer(self._lr)
		trainable_vars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 5)
		self.train_op = self.optimizer.apply_gradients(zip(grads, trainable_vars), global_step=self.global_step)
		
		return self.nodes

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	def get_init_ops(self):
		#init_ops = [tf.initialize_all_variables()]
		init_ops = [tf.global_variables_initializer()]
		voc_file = open(self.voc_file_path, 'r')
		keys = [line.strip() for line in voc_file]
		values = [i for i in range(len(keys))]
		insert_op = self.w2id_table.insert(tf.constant(keys, dtype=tf.string), tf.constant(values, dtype=tf.int64))
		init_ops.extend([insert_op])
		return init_ops
	

	def evaluate(self, session, eval_set):
		all_y_pred=[]
		all_y_true=[]
		for i in range(eval_set.batch_count):
			ret = eval_set.get_batch(i)
			feed_dict = {
				self.input_x: ret[0],
				self.input_y: ret[1],
				self.dropout_keep_prob: 1.0
			}
			step, loss_value, correct_value, y_pred, y_true = session.run(
				[self.global_step, self.loss, self.correct, self.predictions, self.input_y], feed_dict)
			all_y_pred = np.concatenate((all_y_pred, y_pred))
			all_y_true = np.concatenate((all_y_true, y_true))
		accuracy = metrics.accuracy_score(all_y_true, all_y_pred)
		pos_precision = metrics.precision_score(all_y_true, all_y_pred, pos_label=1, average='binary')
		pos_recall = metrics.recall_score(all_y_true, all_y_pred, pos_label=1, average='binary')
		pos_f1 = metrics.f1_score(all_y_true, all_y_pred, pos_label=1, average='binary')
		neg_precision = metrics.precision_score(all_y_true, all_y_pred, pos_label=0, average='binary')
		neg_recall = metrics.recall_score(all_y_true, all_y_pred, pos_label=0, average='binary')
		neg_f1 = metrics.f1_score(all_y_true, all_y_pred, pos_label=0, average='binary')
		
		return accuracy, pos_precision, pos_recall, neg_precision, neg_recall, neg_f1, pos_f1

	def train(self, session, num_epochs, lr_init, max_decay_epoch, dropout_keep_prob, evaluate_every):
		
		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		self.checkpoint_dir = self.out_dir + "/checkpoints"
		self.checkpoint_prefix = self.checkpoint_dir + "/model"
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		#self.saver = tf.train.Saver(tf.global_variables() + 
		#							tf.get_default_graph().get_collection("w2id_table"))
		self.saver = tf.train.Saver()

		batch_count = self.train_set.batch_count
		best_accuracy = 0.0
		best_p_precision = 0.0
		best_p_recall = 0.0
		best_n_precision = 0.0
		best_n_recall = 0.0
		lr_decay = 0.6
		lr_end = 1e-5
		best_neg_f1 = 0.0
		best_pos_f1 = 0.0

		for epoch in range(num_epochs):
			lr_decay = lr_decay ** max(epoch + 1 - max_decay_epoch, 0.0)
			decay_lr = lr_init * lr_decay
			if decay_lr < lr_end:
				break

			self.assign_lr(session, 1e-3)
			shuffle_indices = np.random.permutation(np.arange(batch_count))

			for i in range(batch_count):
				ret = self.train_set.get_batch(shuffle_indices[i])
				feed_dict = {
					self.input_x: ret[0],
					self.input_y: ret[1],
					self.dropout_keep_prob: dropout_keep_prob
				}

				_, step, loss_value, correct_value = session.run(
					[self.train_op, self.global_step, self.loss, self.correct], feed_dict)

				time_str = datetime.datetime.now().isoformat()
				print("epoch {}, step {}, loss {:g}, correct {:.3f}, dev_acc {:g}, pp {:.3f}, pr {:.3f}, np {:.3f}, nr {:.3f}, neg_f1 {:.4f}, pos_f1 {:.4f}"
					.format(epoch, step, loss_value, correct_value*1.0/64, best_accuracy, best_p_precision, best_p_recall,
							best_n_precision, best_n_recall, best_neg_f1, best_pos_f1))

				current_step = tf.train.global_step(session, self.global_step)

				if current_step % evaluate_every == 0:
					print("\nEvaluation:")
					accuracy, p_precision, p_recall, n_precision, n_recall, neg_f1, pos_f1 = self.evaluate(session, self.dev_set)
					print('evaluation accuracy = ', accuracy)
					if best_accuracy < accuracy:
					#if best_neg_f1 < neg_f1:
					#if best_pos_f1 < pos_f1:
						best_accuracy = accuracy
						best_p_precision = p_precision
						best_p_recall = p_recall
						best_n_precision = n_precision
						best_n_recall = n_recall
						best_neg_f1 = neg_f1
						best_pos_f1 = pos_f1
						path = self.saver.save(session, self.checkpoint_prefix, global_step=current_step)
						print("Saved model checkpoint to {}\n".format(path))
						print('best model saved!')
						print("\nTest:")

		return best_accuracy, best_p_precision, best_p_recall, best_n_precision, best_n_recall

	def get_saver(self):
		#saver = tf.train.Saver(tf.global_variables())
		var_list = (tf.global_variables() +
					tf.get_default_graph().get_collection("w2id_table"))
		saver = tf.train.Saver(var_list=var_list)
		return saver

	def export(self, sess, export_dir):
		# export the model
		#export_dir = self.out_dir + "/export/455558" # for export the model as Tensorflow Serving format
		#export_dir =  "./export/455558" # for export the model as Tensorflow Serving format
		inputs = {k:tf.saved_model.utils.build_tensor_info(v) for k,v in self.nodes["inputs"].items()}
		outputs = {k:tf.saved_model.utils.build_tensor_info(v) for k,v in self.nodes["outputs"].items()}
		signature_def = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
			method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
		builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
		builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
				signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature_def})
		builder.save()
		print('Exporting trained model to %s' % export_dir)

		return

	def query_pad(self, query_seg):
		if self.text_len > len(query_seg):
			query_seg += [self.pad_word] * (self.text_len - len(query_seg))
		else:
			query_seg = query_seg[0:self.text_len]
		return query_seg

	def preproc(self, queries, use_seg=True):
		# seg: format as [['how','are'], ['who', 'are','you']]
		if use_seg:
			queries = [seg_return_string(q).decode('gbk').encode('utf-8') for q in queries]
		queries_seg = [q.split() for q in queries]
		# pad
		queries_seg_pad = [self.query_pad(q) for q in queries_seg]
		# feed dict
		feed_dict = {self.input_x_name:queries_seg_pad}

		return feed_dict

	def after_proc(self, output, neg_confidence_threshold=0.6):
		#preds = output[self.output_pred_name]
		probs = output[self.output_prob_name]
		preds = np.asarray([prob[1] > (1.0-neg_confidence_threshold) for prob in probs], dtype=np.int32)
		out_after_proc = {
			'tags' : preds,
			'probs' : probs,
			'neg_confidence_threshold' : neg_confidence_threshold
		}
		return out_after_proc


	class KimCNNLayer(object):
		def __init__(self, inputs, sequence_length, embedding_size, filter_sizes, num_filters, dropout_prob):
			inputs_expanded = tf.expand_dims(inputs, -1)
			# Create a convolution + maxpool layer for each filter size
			pooled_outputs = []
			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope("conv-maxpool-%s" % filter_size):
					# Convolution Layer
					filter_shape = [filter_size, embedding_size, 1, num_filters]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
					#b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
					b = tf.Variable(tf.truncated_normal(shape=[num_filters], stddev=0.1), name="b")
					conv = tf.nn.conv2d(
						inputs_expanded,
						W,
						strides=[1, 1, 1, 1],
						padding="VALID",
						name="conv")
					# Apply nonlinearity
					h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
					# Maxpooling over the outputs
					pooled = tf.nn.max_pool(
						h,
						ksize=[1, sequence_length - filter_size + 1, 1, 1],
						strides=[1, 1, 1, 1],
						padding='VALID',
						name="pool")
					pooled_outputs.append(pooled)
		
			# Combine all the pooled features
			num_filters_total = num_filters * len(filter_sizes)
			#self.h_pool = tf.concat(3, pooled_outputs)
			self.h_pool = tf.concat(pooled_outputs, 3)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
			self.outsize = num_filters_total
			
			# Add dropout
			with tf.name_scope("dropout"):
				self.output = tf.nn.dropout(self.h_pool_flat, dropout_prob)


	class EmbeddingWordLayer(object):
		def __init__(self, inputs, word2id, vocab_size, embedding_size, word_vec_file=None, pickle_vec=None):
			with tf.device('/cpu:0'), tf.name_scope("embedding"):
				w2v = np.random.uniform(-1.0, 1.0, (vocab_size+1, embedding_size)) # vocab_size+1 for adding padding word
				w2v = np.cast['float32'](w2v)

				if word_vec_file is not None:
					# gensim format
					w2v = np.random.uniform(-1.0, 1.0, (vocab_size+1, embedding_size)) # vocab_size+1 for adding padding word
					w2v = np.cast['float32'](w2v)
					#model = Word2Vec.load_word2vec_format(word_vec_file, binary=True)
					model = KeyedVectors.load_word2vec_format(word_vec_file, binary=False)
					for word in model.vocab:
						if word in word2id:
							w2v[word2id[word]+1] = model[word] # word2id[word]+1 for adding padding word
				w2v[0] = np.zeros(embedding_size) # this line for adding paddding word
				word_vec = tf.Variable(w2v, name='word_embedding')
				self.output = tf.nn.embedding_lookup(word_vec, inputs)
