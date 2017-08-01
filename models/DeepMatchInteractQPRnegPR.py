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

def FeatureMatrix(conv_conf, inps, scope=None, dtype=tf.float32):
    with variable_scope.variable_scope(scope) as scope: 
        for i, each in enumerate(conv_conf):
            h, w, ci, co = each[0]  
            h_s, w_s = each[1]
            ph, pw = each[2]
            ph_s, pw_s = each[3]
            k = tf.get_variable("filter_%d" % i, [h, w, ci, co], initializer=tf.random_uniform_initializer(-0.4, 0.4))
            conved = tf.nn.conv2d(inps, k, [1, h_s, w_s, 1], padding="SAME")
            #conved = relu(conved)
            conved = tf.nn.tanh(conved)
            # TODO Max pooling (May be Dynamic-k-max-pooling TODO)
            max_pooled = tf.nn.max_pool(value=conved, ksize=[1, ph, pw, 1], strides=[1, ph, pw, 1], data_format="NHWC", padding="SAME") 
            inps = max_pooled
    return inps 

def FC(inputs, h_size, o_size, act):
    fc1 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=h_size, activation_fn=relu,
                                            weights_initializer=tf.random_uniform_initializer(-0.4, 0.4),
                                            biases_initializer=tf.random_uniform_initializer(-0.4, 0.4))
    fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=o_size, activation_fn=act,
                                            weights_initializer=tf.random_uniform_initializer(-0.3, 0.3),
                                            biases_initializer=tf.random_uniform_initializer(-0.4, 0.4))
    return fc2


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
        cells.append(single_cell)
    return MultiRNNCell(cells) 

class DeepMatchInteractQPRnegPR(object):
    def __init__(self, name, for_deploy, job_type="single", dtype=tf.float32, data_dequeue_op=None):
        self.conf = conf = confs[name]
        self.model_kind = self.__class__.__name__
        if conf.model_kind != self.model_kind:
            print "Wrong model kind !, this model needs config of kind '%s', but a '%s' config is given." % (self.model_kind, conf.model_kind)
            exit(0)
        self.embedding = None
        self.for_deploy = for_deploy
        self.global_step = None 
        self.data_dequeue_op = data_dequeue_op
        self.dtype = dtype
        self.name = name
        self.job_type = job_type

    def build(self):
        conf = self.conf
        dtype = self.dtype
        # All possible inputs
        graphlg.info("Creating inputs and tables...")
        batch_size = None 
        self.enc_querys = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_querys")
        self.query_lens = tf.placeholder(tf.int32, shape=[batch_size],name="query_lens")

        self.enc_posts = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_posts")
        self.post_lens = tf.placeholder(tf.int32, shape=[batch_size],name="post_lens")

        self.enc_resps = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_resps")
        self.resp_lens = tf.placeholder(tf.int32, shape=[batch_size],name="resp_lens")

        
        self.enc_neg_resps = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_neg_resps")
        self.neg_resp_lens = tf.placeholder(tf.int32, shape=[batch_size],name="neg_resp_lens")

        self.enc_neg_posts = tf.placeholder(tf.string, shape=[batch_size, conf.input_max_len], name="enc_neg_posts")
        self.neg_post_lens = tf.placeholder(tf.int32, shape=[batch_size],name="neg_post_lens")

        #TODO table obj, lookup ops and embedding and its lookup op should be placed on the same device
        with tf.device("/cpu:0"):
            self.embedding = variable_scope.get_variable("embedding", [conf.input_vocab_size, conf.embedding_size],
                                                            initializer=tf.random_uniform_initializer(-0.08, 0.08))

            self.in_table = lookup.MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64, default_value=UNK_ID,
                                                    shared_name="in_table", name="in_table", checkpoint=True)
            self.query_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_querys))
            self.post_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_posts))
            self.resp_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_resps))

            self.neg_post_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_neg_posts))
            self.neg_resp_embs = embedding_lookup_unique(self.embedding, self.in_table.lookup(self.enc_neg_resps))

        # MultiRNNCell
         
        graphlg.info("Creating multi-layer cells...")

        # Bi-RNN encoder
        graphlg.info("Creating bi-rnn...")
        if self.for_deploy:
            conf.output_keep_prob = 1.0
        with variable_scope.variable_scope("q_rnn", dtype=dtype, reuse=None) as scope: 
            cell1 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            cell2 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            q_out, q_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.query_embs, sequence_length=self.query_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False)
        with variable_scope.variable_scope("p_rnn", dtype=dtype, reuse=None) as scope: 
            cell1 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            cell2 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            p_out, p_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.post_embs, sequence_length=self.post_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False)

        with variable_scope.variable_scope("p_rnn", dtype=dtype, reuse=True) as scope: 
            cell1 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            cell2 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            neg_p_out, neg_p_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.neg_post_embs, sequence_length=self.neg_post_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False)

        with variable_scope.variable_scope("r_rnn", dtype=dtype, reuse=None) as scope: 
            cell1 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            cell2 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
            r_out, r_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.resp_embs, sequence_length=self.resp_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False)

        with variable_scope.variable_scope("r_rnn", dtype=dtype, reuse=True) as scope: 
            cell1 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob, reuse=True)
            cell2 = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob, reuse=True)
            neg_r_out, neg_r_out_state = bidirectional_dynamic_rnn(cell_fw=cell1, cell_bw=cell2,
                                                           inputs=self.neg_resp_embs, sequence_length=self.neg_resp_lens,
                                                           initial_state_fw=None, initial_state_bw=None,
                                                           dtype=dtype, parallel_iterations=16, swap_memory=False,
                                                           time_major=False)
        fw, bw = q_out_state
        q_out_state = tf.concat([fw[-1].h, bw[-1].h], axis=1)

        fw, bw = p_out_state
        p_out_state = tf.concat([fw[-1].h, bw[-1].h], axis=1)
        fw, bw = neg_p_out_state
        neg_p_out_state = tf.concat([fw[-1].h, bw[-1].h], axis=1)

        fw, bw = r_out_state
        r_out_state = tf.concat([fw[-1].h, bw[-1].h], axis=1)
        fw, bw = neg_r_out_state
        neg_r_out_state = tf.concat([fw[-1].h, bw[-1].h], axis=1)

        q_out = tf.concat(q_out, axis=2)
        p_out = tf.concat(p_out, axis=2)
        r_out = tf.concat(r_out, axis=2)

        neg_p_out = tf.concat(neg_p_out, axis=2)
        neg_r_out = tf.concat(neg_r_out, axis=2)



        # Outputs 2-dim intersection
        graphlg.info("Creating cos dist...")
        qp_sim = tf.expand_dims(tf.matmul(q_out, p_out, transpose_b=True), -1)
        pr_sim = tf.expand_dims(tf.matmul(p_out, r_out, transpose_b=True), -1)
        qr_sim = tf.expand_dims(tf.matmul(q_out, r_out, transpose_b=True), -1)

        qnegp_sim = tf.expand_dims(tf.matmul(q_out, neg_p_out, transpose_b=True), -1)
        pnegr_sim = tf.expand_dims(tf.matmul(p_out, neg_r_out, transpose_b=True), -1)
        qnegr_sim = tf.expand_dims(tf.matmul(q_out, neg_r_out, transpose_b=True), -1)

        # n-CNN max-poolling 
        graphlg.info("Creating interactions...")
        with variable_scope.variable_scope("qp_cnn", dtype=dtype, reuse=None) as scope: 
            qp_map = FeatureMatrix(conf.conv_conf, qp_sim, scope=scope, dtype=dtype)

        with variable_scope.variable_scope("pr_cnn", dtype=dtype, reuse=None) as scope: 
            pr_map = FeatureMatrix(conf.conv_conf, pr_sim, scope=scope, dtype=dtype)

        with variable_scope.variable_scope("qr_cnn", dtype=dtype, reuse=None) as scope: 
            qr_map = FeatureMatrix(conf.conv_conf, qr_sim, scope=scope, dtype=dtype)

        # For neg
        with variable_scope.variable_scope("qp_cnn", dtype=dtype, reuse=True) as scope: 
            qnegp_map = FeatureMatrix(conf.conv_conf, qnegp_sim, scope=scope, dtype=dtype)

        with variable_scope.variable_scope("pr_cnn", dtype=dtype, reuse=True) as scope: 
            pnegr_map = FeatureMatrix(conf.conv_conf, pnegr_sim, scope=scope, dtype=dtype)

        with variable_scope.variable_scope("qr_cnn", dtype=dtype, reuse=True) as scope: 
            qnegr_map = FeatureMatrix(conf.conv_conf, qnegr_sim, scope=scope, dtype=dtype)


        # h becomes 1 after max poolling
        qp_vec = tf.concat([tf.contrib.layers.flatten(qp_map), q_out_state, p_out_state], 1)
        pr_vec = tf.concat([tf.contrib.layers.flatten(pr_map), p_out_state, r_out_state], 1)
        qr_vec = tf.concat([tf.contrib.layers.flatten(qr_map), q_out_state, r_out_state], 1)

        # for neg
        qnegp_vec = tf.concat([tf.contrib.layers.flatten(qnegp_map), q_out_state, neg_p_out_state], 1)
        pnegr_vec = tf.concat([tf.contrib.layers.flatten(pnegr_map), p_out_state, neg_r_out_state], 1)
        qnegr_vec = tf.concat([tf.contrib.layers.flatten(qnegr_map), q_out_state, neg_r_out_state], 1)


        graphlg.info("Creating fully connected...")
        with variable_scope.variable_scope("qp_fc", dtype=dtype, reuse=None) as scope: 
            qp_fc = FC(inputs=qp_vec, h_size=conf.fc_h_size, o_size=1, act=tf.nn.sigmoid)

        with variable_scope.variable_scope("pr_fc", dtype=dtype, reuse=None) as scope: 
            pr_fc = FC(inputs=pr_vec, h_size=conf.fc_h_size, o_size=1, act=relu)

        with variable_scope.variable_scope("qr_fc", dtype=dtype, reuse=None) as scope: 
            qr_fc = FC(inputs=qr_vec, h_size=conf.fc_h_size, o_size=1, act=relu)

        # for neg
        with variable_scope.variable_scope("qp_fc", dtype=dtype, reuse=True) as scope: 
            qnegp_fc = FC(inputs=qnegp_vec, h_size=conf.fc_h_size, o_size=1, act=tf.nn.sigmoid)

        with variable_scope.variable_scope("pr_fc", dtype=dtype, reuse=True) as scope: 
            pnegr_fc = FC(inputs=pnegr_vec, h_size=conf.fc_h_size, o_size=1, act=relu)

        with variable_scope.variable_scope("qr_fc", dtype=dtype, reuse=True) as scope: 
            qnegr_fc = FC(inputs=qnegr_vec, h_size=conf.fc_h_size, o_size=1, act=relu)
        
        #self.scores = tf.squeeze(qp_fc * pr_fc + (1 - qp_fc) * qr_fc)
        #self.neg_scores = tf.squeeze(qnegp_fc * pnegr_fc + (1 - qnegp_fc) * qnegr_fc)
        self.scores = tf.squeeze(qp_fc * pr_fc +  qr_fc)
        self.neg_scores = tf.squeeze(qnegp_fc * pnegr_fc +  qnegr_fc)
        
        graphlg.info("Creating optimizer and backpropagation...")
        self.global_params = []
        self.trainable_params = tf.trainable_variables()
        self.optimizer_params = []

        if not self.for_deploy:
            with variable_scope.variable_scope(self.model_kind, dtype=dtype) as scope: 
                #self.loss = tf.losses.hinge_loss(self.neg_scores, self.scores)
                self.loss = tf.reduce_mean(tf.nn.relu(1 + self.neg_scores - self.scores))
                self.summary = tf.summary.scalar("%s/loss" % self.name, self.loss)

            graphlg.info("Creating backpropagation graph and optimizers...")
            self.learning_rate = tf.Variable(float(conf.learning_rate), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * conf.learning_rate_decay_factor)
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
                 "RMSProp":tf.train.RMSPropOptimizer(self.learning_rate)
            }

            self.opt = self.optimizers[conf.opt_name]
            tmp = set(tf.global_variables()) 

            if self.job_type == "worker": 
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
        self.model_exporter = exporter.Exporter(self.saver)
        inputs = {
                "enc_querys:0": self.enc_querys,
                "query_lens:0": self.query_lens,
                "enc_posts:0": self.enc_posts,
                "post_lens:0": self.post_lens,
                "enc_resps:0": self.enc_resps,
                "resp_lens:0": self.resp_lens
        } 
        outputs = {"out":self.scores}
        self.model_exporter.init(
            tf.get_default_graph().as_graph_def(),
            named_graph_signatures={
                "inputs": exporter.generic_signature(inputs),
                "outputs": exporter.generic_signature(outputs)
            }
        )

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

    # Input is a string line
    def fetch_test_data(self, records, begin=0, size=1):
        feed_data = []
        golds = [] 
        for line in records:
            line = line.strip()
            segs = re.split("\t", line)
            #if len(segs) != 3:
            #    continue
            q = tokenize_word(segs[0])
            p = q 
            #p = tokenize_word(segs[1]) 
            #r = tokenize_word(segs[2])
            r = tokenize_word(segs[1])
            feed_data.append([q, len(q) + 1, p, len(p) + 1, r, len(r) + 1, [], 1, [], 1])
            golds.append(segs[-1])
        return feed_data, golds 

    # Input is a string line
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
            segs = re.split("\t", each.strip())
            if len(segs) != 5:
                continue
            q, p, r, neg_p, neg_r = segs

            q_list = re.split(" +", q.strip())
            p_list = re.split(" +", p.strip()) 
            r_list = re.split(" +", r.strip())
            neg_p_list = re.split(" +", neg_p.strip()) 
            neg_r_list = re.split(" +", neg_r.strip())
            data.append((q_list, len(q_list) + 1, p_list, len(p_list) + 1, r_list, len(r_list) + 1,
                            neg_p_list, len(neg_p_list) + 1, neg_r_list, len(neg_r_list) + 1))
        return data 

    # Input is a batch of examples to be preprocessed
    def get_batch(self, examples):
        conf = self.conf
        (enc_querys, query_lens, enc_posts, post_lens, enc_resps, resp_lens,
                    neg_posts, neg_post_lens, neg_resps, neg_resp_lens) = [], [], [], [], [], [], [], [], [], []

        for q, q_len, p, p_len, r, r_len, neg_p, neg_p_len, neg_r, neg_r_len in examples: 
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
                neg_p_len = neg_r_len if neg_r_len < conf.input_max_len else conf.input_max_len
                neg_p = neg_p[0:conf.input_max_len]
                neg_posts.append(neg_p + ["_PAD"] * (conf.input_max_len - len(neg_p)))
                neg_post_lens.append(np.int32(neg_p_len))

                neg_r_len = neg_r_len if neg_r_len < conf.input_max_len else conf.input_max_len
                neg_r = neg_r[0:conf.input_max_len]
                neg_resps.append(neg_r + ["_PAD"] * (conf.input_max_len - len(neg_r)))
                neg_resp_lens.append(np.int32(neg_r_len))

        feed_dict = { 
                "enc_querys:0": enc_querys,
                "query_lens:0": query_lens,

                "enc_posts:0": enc_posts,
                "post_lens:0": post_lens,
                "enc_resps:0": enc_resps,
                "resp_lens:0": resp_lens,

                "enc_neg_posts:0": neg_posts,
                "neg_post_lens:0": neg_post_lens,
                "enc_neg_resps:0": neg_resps,
                "neg_resp_lens:0": neg_resp_lens
        }
        for k, v in feed_dict.items():
            if not v:
                del feed_dict[k]

        return feed_dict 

    def final(self, outputs):
        return outputs

    def step(self, session, input_feed, forward_only=False, debug=False, run_options=None, run_metadata=None):
        if not forward_only and self.for_deploy: 
            graphlg.error("Error ! Model is for deployment, forward_only must be true")
            return None 
    
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
            outs = session.run(output, input_feed, options=run_options, run_metadata=run_metadata)
            graphlg.info("TIME %.3f" % (time.time() - t))
            summary, loss, output = outs[0], outs[1], outs[2]
            return summary, loss, scores 
        else: 
            output = [
                self.scores
            ]
            outs = session.run(output, input_feed, options=run_options, run_metadata=run_metadata)
            return summary, loss, outs[0] 

if __name__ == "__main__":
    with tf.device("/gpu:3"):
        model = DeepMatchInteractQPRnegPR(name="stc-2-interact-qpr-negpr", for_deploy=False)
        model.build()

    init_ops = model.get_init_ops(job_type="single", task_id=0)

    path = os.path.join(confs["stc-2-interact-qpr-negpr"].data_dir, "train.data")
    qr = QueueReader(filename_list=[path])
    deq_batch_record = qr.batched(batch_size=256, min_after_dequeue=2)

    gpu_options = tf.GPUOptions(allow_growth=True)
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
