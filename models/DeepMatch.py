import time
import random
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops import variable_scope


from QueueReader import *
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python import debug as tf_debug

from tensorflow.contrib import lookup
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.sync_replicas_optimizer import SyncReplicasOptimizer

from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique

from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import EmbeddingWrapper, MultiRNNCell, AttentionCellWrapper, GRUCell, LSTMCell, LSTMStateTuple

from dyn_rnn_loss import dyn_sequence_loss
from tensorflow.contrib.session_bundle import exporter

from config import confs
from util import *

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

def FeatureMatrix(conf, rnn_outs, scope=None, dtype=tf.float32):
    with variable_scope.variable_scope(scope) as scope: 
        # CNN_1 
        for_conv = tf.expand_dims(rnn_outs, -1)
        print for_conv
        #tf.contrib.layers.xavier_initializer_conv2d()
        k = tf.get_variable("filter", [conf.m1, 1, 1, conf.c1], initializer=tf.random_uniform_initializer(-0.05, 0.05))
        conved = tf.nn.conv2d(for_conv, k, [1, 1, 1, 1], padding="SAME")
        conved = relu(conved)
        #conved = tf.nn.tanh(conved)

        # Max pooling (May be Dynamic-k-max-pooling TODO)
        h = conf.input_max_len
        # TODO problem here
        max_pooled = tf.nn.max_pool(value=conved, ksize=[1, h, 1, 1], strides=[1, h, 1, 1], data_format="NHWC", padding="SAME") 
        
        # CNN_2
        # Max pooling (May be Dynamic-k-max-pooling TODO)
        # Folding
    return max_pooled

def CreateCell(conf):
    if conf.cell_model == "GRUCell":
            single_cell = GRUCell(num_units=conf.num_units)
    elif conf.cell_model == "LSTMCell":
        single_cell = LSTMCell(conf.num_units, use_peepholes=conf.use_peepholes)
    else:
        graphlg.info("Unknown Cell type !")
        exit(0)
    if conf.output_keep_prob < 1.0:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=conf.output_keep_prob) 
        graphlg.info("Dropout used: output_keep_prob %f" % conf.output_keep_prob)
    return single_cell

class DeepMatch(object):
    def __init__(self, name, for_deploy=False, job_type="single", dtype=tf.float32, data_dequeue_op=None):
        self.conf = conf = confs[name]
        self.model_kind = self.__class__.__name__
        if self.conf.model_kind != self.model_kind:
            print "Wrong model kind !, this model needs config of kind '%s', but a '%s' config is given." % (self.model_kind, self.conf.model_kind)
            exit(0)

        self.embedding = None
        self.global_step = None 
        self.for_deploy = for_deploy
        self.data_dequeue_op = data_dequeue_op

    def build(self):
        # All possible inputs
        graphlg.info("Creating inputs and tables...")
        batch_size = None 
        self.enc_querys = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_querys")
        self.query_lens = tf.placeholder(tf.int32, shape=[batch_size],name="query_lens")

        self.enc_posts = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_posts")
        self.post_lens = tf.placeholder(tf.int32, shape=[batch_size],name="post_lens")

        self.enc_resps = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_resps")
        self.resp_lens = tf.placeholder(tf.int32, shape=[batch_size],name="resp_lens")
        self.target = tf.placeholder(tf.float32, shape=[batch_size], name="target")

        #TODO table obj, lookup ops and embedding and its lookup op should be placed on the same device
        with tf.device("/cpu:0"):
            self.embedding = variable_scope.get_variable("embedding", [conf.input_vocab_size, conf.embedding_size],
                                                            initializer=tf.random_uniform_initializer(-0.08, 0.08))

            self.in_table = lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64, default_value=UNK_ID,
                                                    shared_name="in_table", name="in_table", checkpoint=True)
            self.query_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_querys))
            self.post_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_posts))
            self.resp_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_resps))

        # MultiRNNCell
         
        graphlg.info("Creating multi-layer cells...")

        # Bi-RNN encoder
        graphlg.info("Creating bi-rnn...")
        #q_out = self.query_embs
        with variable_scope.variable_scope("q_rnn", dtype=dtype, reuse=None) as scope: 
            cell1 = MultiRNNCell([CreateCell(conf) for _ in range(conf.num_layers)])
            cell2 = MultiRNNCell([CreateCell(conf) for _ in range(conf.num_layers)])
            q_out, q_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.query_embs, sequence_length=self.query_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False, scope=scope)
        with variable_scope.variable_scope("p_rnn", dtype=dtype, reuse=None) as scope: 
            cell1 = MultiRNNCell([CreateCell(conf) for _ in range(conf.num_layers)])
            cell2 = MultiRNNCell([CreateCell(conf) for _ in range(conf.num_layers)])
            p_out, p_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.post_embs, sequence_length=self.post_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False, scope=scope)
        with variable_scope.variable_scope("r_rnn", dtype=dtype, reuse=None) as scope: 
            cell1 = MultiRNNCell([CreateCell(conf) for _ in range(conf.num_layers)])
            cell2 = MultiRNNCell([CreateCell(conf) for _ in range(conf.num_layers)])
            r_out, r_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.resp_embs, sequence_length=self.resp_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False, scope=scope)

        #q_out_state = tf.concat(q_out_state, axis=1)
        #p_out_state = tf.concat(p_out_state, axis=1)
        #r_out_state = tf.concat(r_out_state, axis=1)
        
        q_out = tf.concat(q_out, axis=2)
        p_out = tf.concat(p_out, axis=2)
        r_out = tf.concat(r_out, axis=2)

        # Three feature matrice
        graphlg.info("Creating three cnn feature matrice and cos dist...")
        with variable_scope.variable_scope("q_cnn1", dtype=dtype, reuse=None) as scope: 
            q_m = FeatureMatrix(conf, q_out, scope=scope, dtype=dtype)
        with variable_scope.variable_scope("p_cnn1", dtype=dtype, reuse=None) as scope: 
            p_m = FeatureMatrix(conf, p_out, scope=scope, dtype=dtype)
        with variable_scope.variable_scope("r_cnn1", dtype=dtype, reuse=None) as scope: 
            r_m = FeatureMatrix(conf, r_out, scope=scope, dtype=dtype)

        graphlg.info("Creating interactions...")
        # h becomes 1 after max poolling
        q_vec = tf.reshape(q_m, [-1, conf.num_units * 1 * 2 * conf.c1])
        #q_vec = tf.reshape(q_m, [-1, 1 * 1 * conf.c1])
        p_vec = tf.reshape(p_m, [-1, conf.num_units * 1 * 2 * conf.c1])
        #p_vec = tf.reshape(p_m, [-1, 1 * 1 * conf.c1])
        r_vec = tf.reshape(r_m, [-1, conf.num_units * 1 * 2 * conf.c1])
        #r_vec = tf.reshape(r_m, [-1, 1 * 1 * conf.c1])

        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_vec), 1, keep_dims=True))
        norm_p = tf.sqrt(tf.reduce_sum(tf.square(p_vec), 1, keep_dims=True))
        norm_r = tf.sqrt(tf.reduce_sum(tf.square(r_vec), 1, keep_dims=True))
        cos_q_p = tf.reduce_sum(q_vec * p_vec, 1, keep_dims=True) / (norm_q * norm_p)
        cos_q_r = tf.reduce_sum(q_vec * r_vec, 1, keep_dims=True) / (norm_q * norm_r)

        qpcos_vec = tf.concat([q_vec, p_vec, cos_q_p], axis=1)
        qrcos_vec = tf.concat([q_vec, r_vec, cos_q_r], axis=1)
        #qpcos_vec = tf.concat([q_vec, p_vec], axis=1)
        #qrcos_vec = tf.concat([q_vec, r_vec], axis=1)

        h_size = int(math.sqrt(conf.num_units * 2 * 1 * conf.c1 * 2 + 1))

        qp_fc1 = tf.contrib.layers.fully_connected(inputs=qpcos_vec, num_outputs=h_size, activation_fn=relu,
                                                weights_initializer=tf.random_uniform_initializer(-0.08, 0.08),
                                                biases_initializer=tf.random_uniform_initializer(-0.08, 0.08))
        qp_fc2 = tf.contrib.layers.fully_connected(inputs=qp_fc1, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                                weights_initializer=tf.random_uniform_initializer(-0.2, 0.2),
                                                biases_initializer=tf.random_uniform_initializer(-0.4, 0.4))

        qr_fc1 = tf.contrib.layers.fully_connected(inputs=qrcos_vec, num_outputs=h_size, activation_fn=relu,
                                                weights_initializer=tf.random_uniform_initializer(-0.08, 0.08),
                                                biases_initializer=tf.random_uniform_initializer(-0.08, 0.08))

        qr_fc2 = tf.contrib.layers.fully_connected(inputs=qr_fc1, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                                weights_initializer=tf.random_uniform_initializer(-0.2, 0.2),
                                                biases_initializer=tf.random_uniform_initializer(-0.4, 0.4))
        
        self.scores = tf.squeeze(qp_fc2 * qr_fc2)
        
        graphlg.info("Creating optimizer and backpropagation...")
        self.global_params = []
        self.trainable_params = tf.global_variables()
        self.optimizer_params = []

        if not for_deploy:
            with variable_scope.variable_scope("deepmatch", dtype=dtype) as scope: 
                self.loss = tf.reduce_mean(tf.square(self.target - self.scores))
                self.summary = tf.summary.scalar("%s/loss" % name, self.loss)

            self.learning_rate = tf.Variable(float(conf.learning_rate), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * conf.learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.data_idx = tf.Variable(0, trainable=False, name="data_idx")
            self.data_idx_inc_op = self.data_idx.assign(self.data_idx + conf.batch_size)

            graphlg.info("Creating backpropagation graph and optimizers...")
            self.optimizers = {
                 "SGD":tf.train.GradientDescentOptimizer(self.learning_rate),
                 "Adadelta":tf.train.AdadeltaOptimizer(self.learning_rate),
                 "Adagrad":tf.train.AdagradOptimizer(self.learning_rate),
                 "AdagradDA":tf.train.AdagradDAOptimizer(self.learning_rate, self.global_step),
                 "Moment":tf.train.MomentumOptimizer(self.learning_rate, 0.9),
                 "Ftrl":tf.train.FtrlOptimizer(self.learning_rate),
                 "RMSProp":tf.train.RMSPropOptimizer(self.learning_rate)
            }

            self.opt = self.optimizers[conf.opt_name]
            tmp = set(tf.global_variables()) 

            if job_type == "worker": 
                self.opt = SyncReplicasOptimizer(self.opt, conf.replicas_to_aggregate, conf.total_num_replicas) 
                grads_and_vars = self.opt.compute_gradients(loss=self.loss) 
                gradients, variables = zip(*grads_and_vars)  
            else:
                gradients = tf.gradients(self.loss, tf.trainable_variables())
                variables = tf.trainable_variables()

            clipped_gradients, self.grad_norm = tf.clip_by_global_norm(gradients, conf.max_gradient_norm)
            self.update = self.opt.apply_gradients(zip(clipped_gradients, variables), self.global_step)

            self.optimizer_params.append(self.learning_rate)
            self.optimizer_params.extend(list(set(tf.global_variables()) - tmp))
            self.global_params.extend([self.global_step, self.data_idx])
            self.saver = tf.train.Saver(max_to_keep=conf.max_to_keep)


    def get_init_ops(self, job_type, task_id):
        init_ops = []
        if self.conf.embedding_init:
            init_ops = [tf.variables_initializer(set(self.optimizer_params + self.global_params + self.trainable_params)- set([self.embedding]))]
            w2v = np.load(self.conf.embedding_init)
            init_ops.append(self.embedding.assign(w2v))
        else:
            init_ops = [tf.variables_initializer(set(self.optimizer_params + self.global_params + self.trainable_params))]

        if not self.for_deploy and task_id == 0:
            vocab_file = filter(lambda x: re.match("vocab\d+\.all", x) != None, os.listdir(self.conf.data_dir))[0]
            f = codecs.open(os.path.join(self.conf.data_dir, vocab_file))
            k = [next(f).strip() for i in range(self.conf.input_vocab_size)]
            v = range(len(k))
            op_in = self.in_table.insert(constant_op.constant(k), constant_op.constant(v, dtype=tf.int64))
            init_ops.extend([op_in])
        return init_ops

    def get_restorer(self):
        restorer = tf.train.Saver(self.global_params + self.trainable_params + self.optimizer_params +
                                    tf.get_default_graph().get_collection("saveable_objects"))
        return restorer

    def get_dequeue_op(self):
        return None

    def fetch_test_data(self, records, begin=0, size=1):
        feed_data = []
        golds = [] 
        for line in records:
            line = line.strip()
            segs = re.split("\t", line)
            if len(segs) != 3:
                continue
            p = util.tokenize_word(segs[0])
            r = util.tokenize_word(segs[1])
            feed_data.append([p, len(p) + 1, p, len(p) + 1, r, len(r) + 1, 0])
            golds.append(segs[2])
        return feed_data, golds 

    def fetch_train_data(self, records, use_random=False, begin=0, size=1):
        examples = []
        begin = begin % len(records)
        size = len(records) - begin if begin + size >= len(records) else size 
        if use_random == True:
            examples = random.sample(records, size)
        else:
            examples = records[begin:size]
        data = []

        for each in examples:
            p, r, tar = re.split("\t", each.strip())
            p_list = re.split(" +", p.strip())
            q_list = re.split(" +", p.strip()) 
            r_list = re.split(" +", r.strip())
            data.append((q_list, len(q_list) + 1, p_list, len(p_list) + 1, r_list, len(r_list) + 1, np.int32(tar)))
        return data 

    def get_batch(self, examples):
        conf = self.conf
        enc_querys, query_lens, enc_posts, post_lens, enc_resps, resp_lens, tars = [], [], [], [], [], [], []
        for q, q_len, p, p_len, r, r_len, tar in examples: 
            q_len = q_len if q_len < conf.input_max_len else conf.input_max_len
            q = q[0:conf.input_max_len]
            enc_querys.append(q + ["_PAD"] * (conf.input_max_len - len(q)))
            query_lens.append(np.int32(q_len))

            p_len = p_len if p_len < conf.input_max_len else conf.input_max_len
            p = p[0:conf.input_max_len]
            enc_posts.append(p + ["_PAD"] * (conf.input_max_len - len(p)))
            post_lens.append(np.int32(p_len))

            r_len = r_len if r_len < conf.input_max_len else conf.input_max_len
            r = r[0:conf.input_max_len]
            enc_resps.append(r + ["_PAD"] * (conf.input_max_len - len(r)))
            resp_lens.append(np.int32(r_len))
            if not self.for_deploy:
               tars.append(np.float32(tar)) 

        feed_dict = { 
                self.enc_querys.name: enc_querys,
                self.query_lens.name: query_lens,
                self.enc_posts.name: enc_posts,
                self.post_lens.name: post_lens,
                self.enc_resps.name: enc_resps,
                self.resp_lens.name: resp_lens,
                self.target.name: tars,
        }

        return feed_dict 


    def step(self, session, feed_dict, forward_only, debug=False, run_options=None, run_metadata=None):

        loss = None
        summary = None
        scores = None

        if not self.for_deploy:
            if forward_only:
                output = [
                    self.summary,
                    self.loss,
                    self.scores
                ]
            else:
                output = [
                    self.summary,
                    self.loss,
                    self.scores,
                    self.update
                ]
            t = time.time() 
            outs = session.run(output, feed_dict, options=run_options, run_metadata=run_metadata)
            graphlg.info("TIME %.3f" % (time.time() - t))
            summary, loss, output = outs[0], outs[1], outs[2]
            return summary, loss, scores 
        else: 
            output = [
                self.scores
            ]
            outs = session.run(output, feed_dict, options=run_options, run_metadata=run_metadata)
            return summary, loss, outs[0] 

    
if __name__ == "__main__":
    with tf.device("/gpu:2"):
        model = DeepMatch(name="stc-2", for_deploy=False)

    init_ops = model.get_init_ops(job_type="single", task_id=0)

    path = os.path.join(confs["stc-2"].data_dir, "train.data")
    qr = QueueReader(filename_list=[path])
    deq_batch_record = qr.batched(batch_size=1024, min_after_dequeue=2)

    gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options=gpu_options,
                                    intra_op_parallelism_threads=16)

    sess = tf.Session(config=session_config)
    qr.start(session=sess)
    sess.run(init_ops)

    # begin
    batch_record = sess.run(deq_batch_record)
    examples = model.fetch_train_data(batch_record, use_random=False, begin=0, size=len(batch_record))
    feed_dict = model.get_batch(examples)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    while True:
        output = model.step(sess, feed_dict, False)
        print "%.10f" % output[1]
