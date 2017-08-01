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
from CVAERNN import *
from AllAttn import *
from RNNClassification import *

from Tsinghua_plan import *
from Nick_plan import *

from QueueReader import *
from config import *
from util import *

FLAGS = tf.app.flags.FLAGS

# get graph log
graphlg = log.getLogger("graph")

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
        "CVAERNN": CVAERNN,
        "RNNClassification": RNNClassification,
        "Postprob": Postprob,
        "Tsinghua": Tsinghua,
        "AllAttn": AllAttn 
}

def create(conf_name, job_type="single", task_id="0", dtype=tf.float32):
    return magic[confs[conf_name].model_kind](conf_name, job_type, task_id, dtype)

def create_runtime_model(conf_name, for_deploy, ckpt_dir, gpu="", job_type="single", task_id=0, dtype=tf.float32):

    gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                    gpu_options=gpu_options, intra_op_parallelism_threads=16)

    conf = confs[conf_name]
    # Device setting
    def _load_fn(unused_op):
        return 1

    core_str = "cpu:0" if (gpu is None or gpu == "") else "gpu:%d" % int(gpu)
    if job_type == "worker":
        ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(3,_load_fn)
        device = tf.train.replica_device_setter(cluster=conf.cluster,
                                            worker_device='job:worker/task:%d/%s' % (task_id, core_str),
                                            ps_device='job:ps/task:%d/cpu:0' % task_id,
                                            ps_strategy=ps_strategy)
        queue_device = "job:worker/task:0/cpu:0" 
    else:
        device = "/" + core_str 
        queue_device = "/cpu:0" 

    deq_batch_record = None
    path = os.path.join(conf.data_dir, "train.data")
    if conf.use_data_queue:
        with tf.device(queue_device):
            qr = QueueReader(filename_list=[path])
            deq_batch_record = qr.batched(batch_size=conf.batch_size, min_after_dequeue=conf.replicas_to_aggregate)
    

    graph = magic[conf.model_kind](conf_name, for_deploy, job_type, dtype)
    with tf.device(device):
        # Create graph.
        graph.build()
        devices = {}
        for each in tf.trainable_variables():
            if each.device not in devices:
                devices[each.device] = []
            graphlg.info("%s, %s" % (each.name, each.get_shape()))
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
            mem.append("Device: %s, Param size: %s MB" % (d, tmp * dtype.size / 1024.0 / 1024.0))
        graphlg.info(" ========== Device Params Mem ==========")
        for each in mem:
            graphlg.info(each)

    init_ops = graph.get_init_ops(job_type, task_id)

    def init_fn(sess):
        graphlg.info("Saver not used, created model with fresh parameters.")
        print ("initialize new models")
        for each in init_ops:
            graphlg.info("initialize op: %s" % str(each))
            sess.run(each)

    restorer = graph.get_restorer() 
    
    if job_type == "worker":
        server = tf.train.Server(conf.cluster, job_name=job_type, task_index=task_id, config=session_config)
        master = server.target
        ready_for_local_init_op = graph.opt.ready_for_local_init_op
        local_op = graph.opt.chief_init_op if task_id==0 else graph.opt.local_step_init_op
        init_token_op = graph.opt.get_init_tokens_op() if task_id==0 else None
        chief_queue_runner = graph.opt.get_chief_queue_runner() if task_id==0 else None
    else:
        master = ""
        local_op = None
        ready_for_local_init_op = None
        init_token_op = None
        chief_queue_runner = None

    #op = graph.learning_rate.assign(conf.learning_rate)
    sv = tf.train.Supervisor(is_chief=(task_id==0), global_step=graph.global_step,
                            init_op=None, logdir=ckpt_dir, saver=restorer,
                            init_fn=init_fn,
                            ready_for_local_init_op=ready_for_local_init_op,
                            local_init_op=local_op,
                            summary_writer=None,
                            save_model_secs=0)

    sess = sv.prepare_or_wait_for_session(master=master, config=session_config) 
    #sess.run(op)

    if conf.use_data_queue and task_id == 0:
        graphlg.info("chief worker start data queue runners...")
        qr.start(session=sess)

    if job_type == "worker" and task_id == 0:
        graphlg.info("chief worker start parameter queue runners...")
        sv.start_queue_runners(sess, [chief_queue_runner])
        graphlg.info("chief worker insert init tokens...")
        sess.run(init_token_op)

    graphlg.info ("%s:%d Session created" % (job_type, task_id))
    graphlg.info ("Initialization done")

    return graph, sv, sess, deq_batch_record
