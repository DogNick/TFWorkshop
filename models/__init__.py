import logging as log
import codecs
import re
import os
import numpy as np

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import flags
from tensorflow.python.ops import variable_scope

#from DeepMatch import * 
#from DeepMatchContrastive import *
#from DeepMatchInteract import *
#from DeepMatchInteractConcatRNN import *
#from DeepMatchInteractQPPR import *
#from DeepMatchInteractQPRnegPR import *

#from DynAttnTopicSeq2Seq import *
from AttnSeq2Seq import *
from VAERNN import * 
from VAERNN2 import * 
from VAERNN3 import * 
from CVAERNN import *
from AllAttn import *
from KimCNN import *
from RNNClassification import *

from Tsinghua_plan import *
from Nick_plan import *

from QueueReader import *
from config import *
from util import *

FLAGS = tf.app.flags.FLAGS

# get graph log
graphlg = log.getLogger("graph")
trainlg = log.getLogger("train")

magic = {
        #"DeepMatch": DeepMatch,
        #"DeepMatchContrastive": DeepMatchContrastive,
        #"DeepMatchInteract": DeepMatchInteract,
        #"DeepMatchInteractQPPR": DeepMatchInteractQPPR,
        #"DeepMatchInteractQPRnegPR": DeepMatchInteractQPRnegPR,
        #"DeepMatchInteractConcatRNN": DeepMatchInteractConcatRNN,

        #"DynAttnTopicSeq2Seq": DynAttnTopicSeq2Seq,
        "AttnSeq2Seq": AttnSeq2Seq,
        "VAERNN": VAERNN,
        "VAERNN2": VAERNN2,
        "VAERNN3": VAERNN3,
        "CVAERNN": CVAERNN,
        "RNNClassification": RNNClassification,
        "Postprob": Postprob,
        "Tsinghua": Tsinghua,
        "AllAttn": AllAttn,
        "KimCNN":KimCNN
}

def create(conf_name, job_type="single", task_id="0", dtype=tf.float32):
    return magic[confs[conf_name].model_kind](conf_name, job_type, task_id, dtype)




def init_monitored_train(runtime_root, model_core, gpu=""):  
	ckpt_dir = os.path.join(runtime_root, model_core.name)
	conf = model_core.conf
	if not os.path.exists(ckpt_dir):
		os.mkdir(ckpt_dir)

	# create graph logger
	fh = log.FileHandler(os.path.join(ckpt_dir, "graph_%s_%d.log" % (model_core.job_type, model_core.task_id)))
	fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
	log.getLogger("graph").addHandler(fh)
	log.getLogger("graph").setLevel(log.DEBUG) 

	# create training logger
	fh = log.FileHandler(os.path.join(ckpt_dir, "train_%s_%d.log" % (model_core.job_type, model_core.task_id)))
	fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
	log.getLogger("train").addHandler(fh)
	log.getLogger("train").setLevel(log.DEBUG)
	
	#gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
								gpu_options=gpu_options, intra_op_parallelism_threads=32)

	# handle device placement for both single and distributed method
	core_str = "cpu:0" if (gpu is None or gpu == "") else "gpu:%d" % int(gpu)
	if model_core.job_type == "worker":
		def _load_fn(unused_op):
			return 1
		ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(3,_load_fn)
		device = tf.train.replica_device_setter(cluster=conf.cluster,
						worker_device='job:worker/task:%d/%s' % (model_core.task_id, core_str),
						ps_device='job:ps/task:%d/cpu:0' % model_core.task_id,
						ps_strategy=ps_strategy)
		queue_device = "job:worker/task:0/cpu:0"
	else:
		device = "/" + core_str 
		queue_device = "/cpu:0" 

	# Prepare data
	# create a data queue or just read all to memory, attached these data to model object
	path = os.path.join(conf.data_dir, "train.data")
	if conf.use_data_queue:
		with tf.device(queue_device):
			qr = QueueReader(filename_list=[path])
			model_core.dequeue_data_op = self.qr.batched(batch_size=conf.batch_size,
												min_after_dequeue=conf.replicas_to_aggregate)
	else:
		model_core.train_set = []
		model_core.dev_set = []
		count = 0
		path = os.path.join(conf.data_dir, "train.data")
		with codecs.open(path) as f:
			for line in f:
				model_core.train_set.append(line.strip())
				count += 1
				if count % 100000 == 0:
					trainlg.info(" Reading data %d..." % count)
			trainlg.info(" Reading data %d..." % count)
		with codecs.open(os.path.join(conf.data_dir, "valid.data")) as f:
			model_core.dev_set = [line.strip() for line in f]	 

	# Build graph on device
	graph_nodes = model_core.build_all(for_deploy=False, variants="", device=device)

	# Create hooks and master server descriptor	
	saver_hook = hook.NickCheckpointSaverHook(checkpoint_dir=ckpt_dir,
											checkpoint_steps=200,
											model_core=model_core,
											dev_fetches={"loss":graph_nodes["loss"]},
											firein_steps=10000)
	sum_hook = hook.NickSummaryHook(graph_nodes["summary"],
									graph_nodes["debug_outputs"],
									summary_dir=ckpt_dir,
									summary_steps=20,
									debug_steps=40)
	if model_core.job_type == "worker":
		ready_for_local_init_op = model_core.opt.ready_for_local_init_op
		local_op = model_core.opt.chief_init_op if model_core.task_id==0 else model_core.opt.local_step_init_op
		sync_replicas_hook = model_core.opt.make_session_run_hook((model_core.task_id==0))
		master = tf.train.Server(model_core.conf.cluster, job_name=model_core.job_type,
								 task_index=model_core.task_id, config=sess_config,
								 protocol="grpc+verbs").target
		hooks = [sync_replicas_hook, saver_hook, sum_hook]
	else:
		ready_for_local_init_op = None
		local_op = None
		master = ""
		hooks = [saver_hook, sum_hook] 

	scaffold = tf.train.Scaffold(init_op=None, init_feed_dict=None, init_fn=model_core.init_fn(),
								 ready_op=None, ready_for_local_init_op=ready_for_local_init_op,
								 local_init_op=local_op, summary_op=None,
								 saver=model_core.get_restorer())
	sess = tf.train.MonitoredTrainingSession(master=master, is_chief=(model_core.task_id==0),
											checkpoint_dir=ckpt_dir,
											scaffold=scaffold,
											hooks=hooks,
											config=sess_config,
											stop_grace_period_secs=120)
	
	if model_core.conf.use_data_queue and model_core.task_id == 0:
		graphlg.info("chief worker start data queue runners...")
		self.qr.start(session=sess)

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
	return sess, graph_nodes

def init_dummy_train(runtime_root, model_core, create_new=True, gpu=0):
	#gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
	model_core.conf.use_data_queue = True
	conf = model_core.conf
	gpu_options = tf.GPUOptions(allow_growth=True)
	session_config = tf.ConfigProto(allow_soft_placement=True,
									log_device_placement=False,
									gpu_options=gpu_options,
									intra_op_parallelism_threads=32)
	print "Building..."
	graph_nodes = model_core.build_all(for_deploy=False, variants="", device="/gpu:%d" % int(gpu))
	print "Creating Data queue..."
	path = os.path.join(confs[model_core.name].data_dir, "train.data")
	qr = QueueReader(filename_list=[path], shared_name="temp_queue")
	model_core.dequeue_data_op = qr.batched(batch_size=confs[model_core.name].batch_size, min_after_dequeue=3)
	
	if create_new:
		ckpt_dir = os.path.join(runtime_root, "null") 
		if os.path.exists(ckpt_dir):
			shutil.rmtree(ckpt_dir)
	else:
		ckpt_dir = os.path.join(runtime_root, model_core.name)
	scaffold = tf.train.Scaffold(init_op=None, init_feed_dict=None, init_fn=model_core.init_fn(),
								ready_op=None, ready_for_local_init_op=None, local_init_op=None,
								summary_op=None, saver=model_core.get_restorer())
	sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
								gpu_options=gpu_options, intra_op_parallelism_threads=32)
	sess = tf.train.MonitoredTrainingSession(master="", is_chief=True, checkpoint_dir=ckpt_dir, 
								scaffold=scaffold, save_summaries_steps=None, save_summaries_secs=None,
								config=sess_config) 
	#sess = tf.Session(config=sess_config)
	#sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
	#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
	return sess, graph_nodes

def init_inference(runtime_root, model_core, variants="", gpu="", ckpt_steps=None):
	core_str = "cpu:0" if (gpu is None or gpu == "") else "/gpu:%d" % int(gpu)
	ckpt_dir = os.path.join(runtime_root, model_core.name)
	if not os.path.exists(ckpt_dir):
		print ("\n No checkpoint dir found !!! exit")
		exit(0)
	gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
	session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
								gpu_options=gpu_options, intra_op_parallelism_threads=32)
	graph_nodes = model_core.build_all(for_deploy=True, variants=variants, device=core_str)
	sess = tf.Session(config=session_config)
	#self.sess = tf.InteractiveSession(config=session_config)
	#self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

	restorer = model_core.get_restorer()
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
	restorer.restore(save_path=filename, sess=sess)

	return sess, graph_nodes, ckpt_steps
	
def join_param_server(cluster, job_name, task_id):
	gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
	session_config = tf.ConfigProto(allow_soft_placement=True,
								log_device_placement=False,
								gpu_options=gpu_options,
								intra_op_parallelism_threads=32)
	server = tf.train.Server(cluster, job_name=job_type,
							task_index=task_id, config=session_config, protocol="grpc+verbs")
	trainlg.info("ps join...")
	server.join()

