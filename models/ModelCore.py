import sys
sys.path.insert(0, "/search/odin/Nick/_python_build")
import abc
import time
import shutil
import logging as log
from util import * 
from config import confs

from QueueReader import *
import tensorflow as tf

from tensorflow.python.ops import variable_scope
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import signature_constants 
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import main_op
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique
from tensorflow.contrib.tensorboard.plugins import projector

from nick_tf import score_decoder 
from nick_tf import helper
from nick_tf import basic_decoder
from nick_tf import decoder
from nick_tf import beam_decoder
from nick_tf import dynamic_attention_wrapper
from nick_tf.cocob_optimizer import COCOB 

import hook 

graphlg = log.getLogger("graph")
trainlg = log.getLogger("train")

class ModelCore(object):
	def __init__(self, name, job_type="single", task_id="0", dtype=tf.float32):
		self.conf = confs[name]
		self.model_kind = self.__class__.__name__
		if self.conf.model_kind !=  self.model_kind:
			print "Wrong model kind !, this model needs config of kind '%s', but a '%s' config is given." % (self.model_kind, self.conf.model_kind)
			exit(0)

		self.name = name
		self.job_type = job_type
		self.task_id = int(task_id)
		self.dtype = dtype

		# data stub 
		self.sess = None
		self.train_set = []
		self.dev_set = []
		self.dequeue_data_op = None 
		self.prev_max_dev_loss = 10000000
		self.latest_train_losses = []

		self.learning_rate = None
		self.learning_rate_decay_op = None

		self.global_step = None 
		self.trainable_params = []
		self.global_params = []
		self.optimizer_params = []
		self.need_init = []
		self.saver = None
		self.builder = None

		self.run_options = None #tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		self.run_metadata = None #tf.RunMetadata()

	@abc.abstractmethod 
	def build(self, for_deploy, variants=""):
		"""build graph in deploy/train way

		build a graph with two seperate graph branches for both training and inference,
		this function will return a graph_nodes dict in which some specific keys
		are offered.
		
		Params:
			for_deploy: to specify the current branch to build(train or deploy)
			variants: a additional specification for model variants,
		Returns:
			graph_nodes: a dict with following keys:
				{
					"loss":...,
					"inputs":...,
					"outputs":...,
					"debug_outputs":...,
					"visualize":...
				}
				Typically, in training paradigm, loss is required, inputs and outputs are none;
				While in infer paradigm, loss is none and inputs and outputs are required;
				visuallize and debug_outputs are always optional.
				Note that in training paradigm, an 'update' key mapped to a backprop op upon
				graph_nodes['loss'] will be also added to graph_nodes after this function is
				called in build_all(...) 
		"""
		return

	@abc.abstractmethod
	def get_init_ops():
		return

	def init_fn(self):
		init_ops = self.get_init_ops()
		def fn(scaffold, sess):
			graphlg.info("Saver not used, created model with fresh parameters.")
			graphlg.info("initialize new models")
			for each in init_ops:
				graphlg.info("initialize op: %s" % str(each))
				sess.run(each)
		return fn

	@abc.abstractmethod
	def get_restorer(self):
		return
	
	@abc.abstractmethod
	def export(self, conf_name, runtime_root="../runtime", deploy_dir="deployments", ckpt_steps=None, variants=""):
		sess, nodes, global_steps = self.init_infer(gpu="0", variants=variants, ckpt_steps=ckpt_steps, runtime_root=runtime_root)

		# global steps as version
		export_dir = os.path.join(os.path.join(deploy_dir, conf_name), str(global_steps))

		if os.path.exists(export_dir):
			print("Removing duplicate: %s" % export_dir)
			shutil.rmtree(export_dir)

		inputs = {k:utils.build_tensor_info(v) for k, v in nodes["inputs"].items()}
		outputs = {k:utils.build_tensor_info(v) for k, v in nodes["outputs"].items()}

		signature_def = signature_def_utils.build_signature_def(inputs=inputs,
				outputs=outputs, 
				method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
		
		builder = saved_model_builder.SavedModelBuilder(export_dir)
		builder.add_meta_graph_and_variables(sess,
				[tag_constants.SERVING],
				signature_def_map={
						signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature_def
					}
				)
		builder.save()
		print('Exporting trained model to %s' % export_dir)
		return

	@abc.abstractmethod
	def after_proc(self, out):
		return {}


	@abc.abstractmethod
	def print_after_proc(self, after_proc):
		outs = after_proc["outputs"]
		for i, each in enumerate(outs):
			for j, s in enumerate(each):
				out_str = " ".join(s)
				#print out_str, outs[0]["probs"][i][j]
				print out_str, after_proc["probs"][i][j] 

	def fetch_data(self, use_random=False, begin=0, size=128, dev=False, sess=None):
		""" General Fetch data process
		"""
		if self.conf.use_data_queue:
			if sess:
				curr_sess = sess
			elif self.sess == None:
				print "FATAL: The model must be initialized first when data queue used !!"  
				exit(0)
			elif self.dequeue_data_op == None:
				print "FATAL: 'use_data_queue' in conf is true but dequeue_data_op is None"
				exit(0)
			else:
				curr_sess = self.sess
			examples = curr_sess.run(self.dequeue_data_op)
		else:
			records = self.dev_set if dev else self.train_set
			if use_random == True:
				examples = random.sample(records, size)
			else:
				begin = begin % len(records)
				examples = records[begin:begin+size]
		return examples

	def build_all(self, for_deploy, variants="", device="/cpu:0"):
		with tf.device(device):
			#with variable_scope.variable_scope(self.model_kind, dtype=tf.float32) as scope: 
			graphlg.info("Building main graph...")	
			graph_nodes = self.build(for_deploy, variants="")
			graphlg.info("Collecting trainable params...")
			self.trainable_params.extend(tf.trainable_variables())
			if not for_deploy:	
				graphlg.info("Creating backpropagation graph and optimizers...")
				graph_nodes["update"] = self.backprop(graph_nodes["loss"])
				graph_nodes["summary"] = tf.summary.merge_all()
				self.saver = tf.train.Saver(max_to_keep=self.conf.max_to_keep)
			if "visualize" not in graph_nodes:
				graph_nodes["visualize"] = None
			graphlg.info("Graph done")
			graphlg.info("")

		# More log info about device placement and params memory
		devices = {}
		for each in tf.trainable_variables():
			if each.device not in devices:
				devices[each.device] = []
			graphlg.info("%s, %s, %s" % (each.name, each.get_shape(), each.device))
			devices[each.device].append(each)
		mem = []
		graphlg.info(" ========== Params placment ==========")
		for d in devices:
			tmp = 0.0
			for each in devices[d]: 
				#graphlg.info("%s, %s, %s" % (d, each.name, each.get_shape()))
				shape = each.get_shape()
				size = 1.0 
				for dim in shape:
					size *= int(dim)
				tmp += size
			mem.append("Device: %s, Param size: %s MB" % (d, tmp * self.dtype.size / 1024.0 / 1024.0))
		graphlg.info(" ========== Device Params Mem ==========")
		for each in mem:
			graphlg.info(each)
		return graph_nodes

	def backprop(self, loss):
		# Backprop graph and optimizers
		conf = self.conf
		dtype = self.dtype
		#with tf.variable_scope(self.model_kind) as scope:
		with tf.name_scope("backprop") as scope:
			self.learning_rate = tf.Variable(float(conf.learning_rate),
									trainable=False, name="learning_rate")
			self.learning_rate_decay_op = self.learning_rate.assign(
						self.learning_rate * conf.learning_rate_decay_factor)
			self.global_step = tf.Variable(0, trainable=False, name="global_step")
			self.data_idx = tf.Variable(0, trainable=False, name="data_idx")
			self.data_idx_inc_op = self.data_idx.assign(self.data_idx + conf.batch_size)

			self.optimizers = {
				"SGD":tf.train.GradientDescentOptimizer(self.learning_rate),
				"Adadelta":tf.train.AdadeltaOptimizer(self.learning_rate),
				"Adagrad":tf.train.AdagradOptimizer(self.learning_rate),
				"AdagradDA":tf.train.AdagradDAOptimizer(self.learning_rate, self.global_step),
				"Moment":tf.train.MomentumOptimizer(self.learning_rate, 0.9),
				"Ftrl":tf.train.FtrlOptimizer(self.learning_rate),
				"RMSProp":tf.train.RMSPropOptimizer(self.learning_rate),
				"Adam":tf.train.AdamOptimizer(self.learning_rate),
				"COCOB":COCOB()
			}

			self.opt = self.optimizers[conf.opt_name]
			tmp = set(tf.global_variables()) 

			if self.job_type == "worker": 
				self.opt = tf.train.SyncReplicasOptimizer(self.opt, conf.replicas_to_aggregate, conf.total_num_replicas) 
				grads_and_vars = self.opt.compute_gradients(loss=loss) 
				gradients, variables = zip(*grads_and_vars)  
			else:
				gradients = tf.gradients(loss, tf.trainable_variables(), aggregation_method=2)
				variables = tf.trainable_variables()

			clipped_gradients, self.grad_norm = tf.clip_by_global_norm(gradients, conf.max_gradient_norm)
			update = self.opt.apply_gradients(zip(clipped_gradients, variables), self.global_step)

			graphlg.info("Collecting optimizer params and global params...")
			self.optimizer_params.append(self.learning_rate)
			self.optimizer_params.extend(list(set(tf.global_variables()) - tmp))
			self.global_params.extend([self.global_step, self.data_idx])
			tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, self.global_step)
		return update

	def preproc(self, records, use_seg=True, for_deploy=False, default_wgt=1.0):
		# parsing
		data = []
		for each in records:
			if for_deploy:
				p = each.strip()
				words, _ = tokenize_word(p) if use_seg else (p.split(), None)
				p_list = words #re.split(" +", p.strip())
				data.append([p_list, len(p_list) + 1, [], 1, 1.0])
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
				data.append([p_list, len(p_list) + 1, r_list, len(r_list) + 1, down_wgts])

		# batching
		conf = self.conf
		batch_enc_inps, batch_dec_inps, batch_enc_lens, batch_dec_lens, batch_down_wgts = [], [], [], [], []
		for encs, enc_len, decs, dec_len, down_wgts in data:
			# Encoder inputs are padded, reversed and then padded to max.
			enc_len = enc_len if enc_len < conf.input_max_len else conf.input_max_len
			encs = encs[0:conf.input_max_len]
			if conf.enc_reverse:
				encs = list(reversed(encs + ["_PAD"] * (enc_len - len(encs))))
			enc_inps = encs + ["_PAD"] * (conf.input_max_len - len(encs))

			batch_enc_inps.append(enc_inps)
			batch_enc_lens.append(np.int32(enc_len))
			if not for_deploy:
				# Decoder inputs with an extra "GO" symbol and "EOS_ID", then padded.
				decs += ["_EOS"]
				decs = decs[0:conf.output_max_len + 1]
				# fit to the max_dec_len
				if dec_len > conf.output_max_len + 1:
					dec_len = conf.output_max_len + 1
				# Merge dec inps and targets 
				batch_dec_inps.append(["_GO"] + decs + ["_PAD"] * (conf.output_max_len + 1 - len(decs)))
				batch_dec_lens.append(np.int32(dec_len))
				batch_down_wgts.append(down_wgts)
		self.curr_input_feed = feed_dict = {
			"inputs/enc_inps:0": batch_enc_inps,
			"inputs/enc_lens:0": batch_enc_lens,
			"inputs/dec_inps:0": batch_dec_inps,
			"inputs/dec_lens:0": batch_dec_lens,
			"inputs/down_wgts:0": batch_down_wgts
		}
		for k, v in feed_dict.items():
			if not v: 
				del feed_dict[k]
		return feed_dict

	def join_param_server(self):
		gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
		session_config = tf.ConfigProto(allow_soft_placement=True,
									log_device_placement=False,
									gpu_options=gpu_options,
									intra_op_parallelism_threads=32)
		server = tf.train.Server(self.conf.cluster, job_name=self.job_type,
								task_index=self.task_id, config=session_config, protocol="grpc+verbs")
		trainlg.info("ps join...")
		server.join()

	def init_infer(self, gpu="", variants="", ckpt_steps=None, runtime_root="../runtime_root"):
		core_str = "cpu:0" if (gpu is None or gpu == "") else "/gpu:%d" % int(gpu)
		ckpt_dir = os.path.join(runtime_root, self.name)
		print ckpt_dir
		if not os.path.exists(ckpt_dir):
			print ("\n No checkpoint dir found !!! exit")
			exit(0)
		gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
		session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
									gpu_options=gpu_options, intra_op_parallelism_threads=32)
		graph_nodes = self.build_all(for_deploy=True, variants=variants, device=core_str)
		self.sess = tf.Session(config=session_config)
		#self.sess = tf.InteractiveSession(config=session_config)
		#self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

		restorer = self.get_restorer()
		ckpt = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=None)
		filename = None
		if ckpt_steps:
			for each in ckpt.all_model_checkpoint_paths:
				seged = re.split("\-", each)	
				if str(ckpt_steps) == seged[-1]:
					filename = each
					break
		else:
			filename = ckpt.model_checkpoint_path
			ckpt_steps = re.split("\-", filename)[-1]
			print ("use latest %s as inference model" % ckpt_steps)
		if filename == None:
			print ("\n No checkpoint step %s found in %s" % (str(ckpt_steps), ckpt_dir))
			exit(0)
		restorer.restore(save_path=filename, sess=self.sess)
		return self.sess, graph_nodes, ckpt_steps

	def init_monitored_train(self, runtime_root, gpu=""):  
		self.ckpt_dir = os.path.join(runtime_root, self.name)
		if not os.path.exists(self.ckpt_dir):
			os.mkdir(self.ckpt_dir)

		# create graph logger
		fh = log.FileHandler(os.path.join(self.ckpt_dir, "graph_%s_%d.log" % (self.job_type, self.task_id)))
		fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
		log.getLogger("graph").addHandler(fh)
		log.getLogger("graph").setLevel(log.DEBUG) 

		# create training logger
		fh = log.FileHandler(os.path.join(self.ckpt_dir, "train_%s_%d.log" % (self.job_type, self.task_id)))
		fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
		log.getLogger("train").addHandler(fh)
		log.getLogger("train").setLevel(log.DEBUG)
		
		#gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
		gpu_options = tf.GPUOptions(allow_growth=True)
		sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
									gpu_options=gpu_options, intra_op_parallelism_threads=32)

		# handle device placement for both single and distributed method
		core_str = "cpu:0" if (gpu is None or gpu == "") else "gpu:%d" % int(gpu)
		if self.job_type == "worker":
			def _load_fn(unused_op):
				return 1
			ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(3,_load_fn)
			device = tf.train.replica_device_setter(cluster=self.conf.cluster,
							worker_device='job:worker/task:%d/%s' % (self.task_id, core_str),
							ps_device='job:ps/task:%d/cpu:0' % self.task_id,
							ps_strategy=ps_strategy)
			queue_device = "job:worker/task:0/cpu:0"
		else:
			device = "/" + core_str 
			queue_device = "/cpu:0" 

		# Prepare data
		# create a data queue or just read all to memory
		self.train_set = []
		self.dev_set = []
		path = os.path.join(self.conf.data_dir, "train.data")
		if self.conf.use_data_queue:
			with tf.device(queue_device):
				self.qr = QueueReader(filename_list=[path])
				self.dequeue_data_op = self.qr.batched(batch_size=self.conf.batch_size,
													min_after_dequeue=self.conf.replicas_to_aggregate)
		else:
			count = 0
			path = os.path.join(self.conf.data_dir, "train.data")
			with codecs.open(path) as f:
				for line in f:
					self.train_set.append(line.strip())
					count += 1
					if count % 100000 == 0:
						trainlg.info(" Reading data %d..." % count)
				trainlg.info(" Reading data %d..." % count)
			with codecs.open(os.path.join(self.conf.data_dir, "valid.data")) as f:
				self.dev_set = [line.strip() for line in f]	 

		# build all graph on device
		graph_nodes = self.build_all(for_deploy=False, variants="", device=device)

		# Create hooks and master server descriptor	
		saver_hook = hook.NickCheckpointSaverHook(checkpoint_dir=self.ckpt_dir, checkpoint_steps=200,
												  saver=self.saver,
												  fetch_data_fn=self.fetch_data,
												  preproc_fn=self.preproc,
												  dev_fetches={"loss":graph_nodes["loss"]},
												  firein_steps=10000)
		sum_hook = hook.NickSummaryHook(graph_nodes["summary"], graph_nodes["debug_outputs"], self.ckpt_dir, 20, 40)
		if self.job_type == "worker":
			ready_for_local_init_op = self.opt.ready_for_local_init_op
			local_op = self.opt.chief_init_op if self.task_id==0 else self.opt.local_step_init_op
			sync_replicas_hook = self.opt.make_session_run_hook((self.task_id==0))
			master = tf.train.Server(self.conf.cluster, job_name=self.job_type,
									 task_index=self.task_id, config=sess_config,
									 protocol="grpc+verbs").target
			hooks = [sync_replicas_hook, saver_hook, sum_hook]
		else:
			ready_for_local_init_op = None
			local_op = None
			master = ""
			hooks = [saver_hook, sum_hook] 

		scaffold = tf.train.Scaffold(init_op=None, init_feed_dict=None, init_fn=self.init_fn(),
									 ready_op=None, ready_for_local_init_op=ready_for_local_init_op,
									 local_init_op=local_op, summary_op=None,
									 saver=self.get_restorer())
		
		self.sess = tf.train.MonitoredTrainingSession(master=master, is_chief=(self.task_id==0),
													  checkpoint_dir=self.ckpt_dir,
													  scaffold=scaffold,
													  hooks=hooks,
													  config=sess_config,
													  stop_grace_period_secs=120)
		
		if self.conf.use_data_queue and self.task_id == 0:
			graphlg.info("chief worker start data queue runners...")
			self.qr.start(session=self.sess)

		#if self.task_id == 0:
		#	trainlg.info("preparing for summaries...")
		#	sub_dir_train = "summary/%s/%s" % (self.conf.model_kind, self.name) 
		#	self.summary_writer = tf.summary.FileWriter(os.path.join(runtime_root, sub_dir_train), self.sess.graph, flush_secs=0)

		#if self.job_type == "worker" and self.task_id == 0:
		#	#graphlg.info("chief worker start parameter queue runners...")
		#	#sv.start_queue_runners(self.sess, [chief_queue_runner])
		#	graphlg.info("chief worker insert init tokens...")
		#	self.sess.run(init_token_op)
		#	graphlg.info ("%s:%d Session created" % (self.job_type, self.task_id))
		#	graphlg.info ("Initialization done")
		return self.sess, graph_nodes

	def adjust_lr_rate(self, global_step, step_loss):
		self.latest_train_losses.append(step_loss)
		if step_loss < self.latest_train_losses[0]:
			self.latest_train_losses = self.latest_train_losses[-1:] 
		if global_step > self.conf.lr_keep_steps and len(self.latest_train_losses) == self.conf.lr_check_steps:
			self.sess.run(self.learning_rate_decay_op)
			self.latest_train_losses = []

	def get_visual_tensor(self):
		return None, None

	def visualize(self, train_root, gpu=0, records=[], ckpt_steps=None): 
		sess, graph_nodes, global_steps = self.init_infer(gpu=gpu, variants="", ckpt_steps=None, runtime_root=train_root)

		if "visualize" not in graph_nodes:
			print "visualize nodes not found"
			return

		# tf variables and temp variables to hold embs
		embs = {}
		emb_vars = {}
		for start in range(0, len(records), self.conf.batch_size):
			print "Runing examples %d - %d..." % (start, start + self.conf.batch_size)
			batch = records[start:start + self.conf.batch_size]
			input_feed = self.preproc(batch, use_seg=False, for_deploy=True)
			#visuals, outputs = sess.run([graph_nodes["visualize"], graph_nodes["outputs"]], feed_dict=input_feed)
			visuals = sess.run(graph_nodes["visualize"], feed_dict=input_feed)
			for k, v in visuals.items():
				if k not in embs:
					embs[k] = []
				embs[k].append(v)
		for k,v in graph_nodes["visualize"].items():
			dim_size = int(tf.contrib.layers.flatten(v).get_shape()[1])
			emb_vars[k] = tf.Variable(tf.random_normal([len(records), dim_size]), name=k)

		ckpt_dir = os.path.join(train_root, self.name)
		#emb_dir = ckpt_dir + "-embs" 
		emb_dir = os.path.join(ckpt_dir, "embeddings")
		#meta_path = os.path.join(ckpt_dir, "metadata.tsv")
		#meta_path = os.path.join(train_root, "metadata.tsv")


		# do embedding
		meta_path = os.path.join(emb_dir, "metadata.tsv")
		config = projector.ProjectorConfig()
		#summary_writer = tf.summary.FileWriter(os.path.join(train_root, self.name))
		summary_writer = tf.summary.FileWriter(emb_dir, sess.graph)
		print "all keys: %s" % str(embs.keys())
		for node_name, emb_list in embs.items():
			print "Embedding %s..." % node_name
			## may not be used
			#outs = np.concatenate(out_list, axis=0)
			sess.run(emb_vars[node_name].assign(np.concatenate(emb_list, axis=0)))

			embedding = config.embeddings.add()
			embedding.tensor_name = emb_vars[node_name].name 
			embedding.metadata_path =  "metadata.tsv" 

		#saver = tf.train.Saver(emb_vars.values())
		saver = tf.train.Saver(emb_vars.values())
		#saver.save(sess, os.path.join(ckpt_dir, "embeddings"), 0)
		#saver.save(sess, os.path.join(train_root, "embs"), 0)
		saver.save(sess, os.path.join(emb_dir, "embs.ckpt"), 0)
		projector.visualize_embeddings(summary_writer, config)

		# writing metadata
		print "Writing meta data %s..." % meta_path
		with codecs.open(meta_path, "w") as f:
			f.write("Query\tFrequency\n")
			#for i, each in enumerate(outs):
			#	each = list(each)
			#	if "_EOS" in each:
			#		each = each[0:each.index("_EOS")]
			#	f.write("%s --> %s\t%d\n" % (records[i], "".join(each), i))
			for i, line in enumerate(records):
				f.write("%s\t%d\n" % (line, 1))
		return
		
	def dummy_train(self, gpu=0, create_new=True, train_root="../runtime"):
		#gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
		gpu_options = tf.GPUOptions(allow_growth=True)
		session_config = tf.ConfigProto(allow_soft_placement=True,
										log_device_placement=False,
										gpu_options=gpu_options,
										intra_op_parallelism_threads=32)
		print "Building..."
		graph_nodes = self.build_all(for_deploy=False, variants="", device="/gpu:%d" % gpu)
		

		print "Creating Data queue..."
		path = os.path.join(confs[self.name].data_dir, "train.data")
		qr = QueueReader(filename_list=[path], shared_name="temp_queue")
		deq_batch_records = qr.batched(batch_size=confs[self.name].batch_size, min_after_dequeue=3)
		
		if create_new:
			ckpt_dir = os.path.join(train_root, "null") 
			if os.path.exists(ckpt_dir):
				shutil.rmtree(ckpt_dir)
		else:
			ckpt_dir = os.path.join(train_root, self.name)
		scaffold = tf.train.Scaffold(init_op=None, init_feed_dict=None, init_fn=self.init_fn(),
									ready_op=None, ready_for_local_init_op=None, local_init_op=None,
									summary_op=None, saver=self.get_restorer())
		sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
									gpu_options=gpu_options, intra_op_parallelism_threads=32)
		sess = tf.train.MonitoredTrainingSession(master="", is_chief=True, checkpoint_dir=ckpt_dir, 
									scaffold=scaffold, save_summaries_steps=None, save_summaries_secs=None,
									config=sess_config) 
		#sess = tf.Session(config=sess_config)
		#sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
		#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

		print "Dequeue one batch..."
		batch_records = sess.run(deq_batch_records)
		N = 10 
		while True:
			print "Step only on one batch..."
			feed_dict = self.preproc(batch_records, use_seg=False, for_deploy=False)
			t0 = time.time()
			fetches = {
				"loss":graph_nodes["loss"],
				"update":graph_nodes["update"],
				"debug_outputs":graph_nodes["debug_outputs"]
			}
			out = sess.run(fetches, feed_dict)
			t = time.time() - t0
			for i in range(N):
				print "=================="
				for key in feed_dict:
					if isinstance(feed_dict[key][i], list):
						print "%s_%d:" % (key, i), " ".join([str(each) for each in feed_dict[key][i]])
					else:
						print "%s_%d:" % (key, i), str(feed_dict[key][i])
				if isinstance(out["debug_outputs"], dict):
					for k,v in out["debug_outputs"].items():
						print ">>> %s_%d:" % (k, i), " ".join(v[i])
				else:
					print ">>> debug_outputs_%d:" % i, " ".join(out["debug_outputs"][i])
			print "TIME: %.4f, LOSS: %.10f" % (t, out["loss"])
			print ""

	def test(self, gpu=0, use_seg=True):
		sess, graph_nodes, global_steps = self.init_infer(gpu="0", runtime_root="../runtime/")
		while True:
			query = raw_input(">>")
			batch_records = [query]
			feed_dict = self.preproc(batch_records, use_seg=use_seg, for_deploy=True)
			out_dict = sess.run(graph_nodes["outputs"], feed_dict) 
			out = self.after_proc(out_dict) 
			self.print_after_proc(out)

	def test_logprob(self, gpu=0, use_seg=True):
		sess, graph_nodes = self.init_infer(gpu="0", variants="score", runtime_root="../runtime/")
		while True:
			post = raw_input("Post >>")
			resp = raw_input("Response >>")
			words, _ = tokenize_word(resp) if use_seg else p.split()
			resp_str = " ".join(words)
			print "Score resp: %s" % resp_str
			batch_records = ["%s\t%s" % (post, resp_str)]
			feed_dict = self.preproc(batch_records, use_seg=True, for_deploy=False)
			out_dict = sess.run(graph_nodes["outputs"], feed_dict)
			prob = out_dict["logprobs"]
			print prob

	def __call__(self, flag="train", use_seg=True, gpu=0):
		if flag == "train":
			self.dummy_train(gpu)
		elif flag == "trainold":
			self.dummy_train(gpu, create_new=False)
		elif flag == "test":
			self.test(gpu, use_seg)
		elif flag == "test_score":
			self.test_logprob(gpu, use_seg)
		else:
			print "Unknown flag: %s" % flag
			exit(0)
