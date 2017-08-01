#from __future__ import absolute_import
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops

from tensorflow.contrib import lookup
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

from tensorflow.python.training.sync_replicas_optimizer import SyncReplicasOptimizer
from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique

from tensorflow.contrib.rnn import  MultiRNNCell, AttentionCellWrapper, GRUCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn

from tensorflow.contrib.seq2seq.python.ops import decoder, helper, basic_decoder
from attn_topic_decoder import AttnTopicDecoder 

from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.session_bundle import exporter
from tensorflow.contrib import seq2seq as seq2seq

from QueueReader import *
from dyn_rnn_loss import dyn_sequence_loss

from config import confs

import sys
sys.path.append("models")
from util import * 

#get graph logger
graphlg = log.getLogger("graph")

def CreateMultiRNNCell(cell_name, num_units, num_layers=1, output_keep_prob=1.0, reuse=False):
    #tf.contrib.training.bucket_by_sequence_length
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

        #single_cell = DeviceWrapper(ResidualWrapper(single_cell), device='/gpu:%d' % i)
        #single_cell = DeviceWrapper(single_cell, device='/gpu:%d' % i)

        cells.append(single_cell)
    return MultiRNNCell(cells) 

class DynAttnTopicSeq2Seq(object):
    def __init__(self, name, for_deploy, job_type="single", dtype=tf.float32, data_dequeue_op=None):
        self.conf = conf = confs[name]
        self.model_kind = self.__class__.__name__
        if self.conf.model_kind !=  self.model_kind:
            print "Wrong model kind !, this model needs config of kind '%s', but a '%s' config is given." % (self.model_kind, self.conf.model_kind)
            exit(0)

        self.for_deploy = for_deploy
        self.embedding = None
        self.out_proj = None
        self.global_step = None 
        self.data_dequeue_op = data_dequeue_op 
        self.name = name
        self.job_type = job_type

        self.trainable_params = []
        self.global_params = []
        self.optimizer_params = []
        self.dtype=dtype

    def build(self):
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
        
        self.topic_in_table = lookup.MutableHashTable(key_dtype=tf.string,
                                                     value_dtype=tf.int64,
                                                     default_value=2,
                                                     shared_name="topic_in_table",
                                                     name="topic_in_table",
                                                     checkpoint=True)

        self.out_table = lookup.MutableHashTable(key_dtype=tf.int64,
                                                 value_dtype=tf.string,
                                                 default_value="_UNK",
                                                 shared_name="out_table",
                                                 name="out_table",
                                                 checkpoint=True)

        graphlg.info("Creating placeholders...")
        self.enc_str_inps = tf.placeholder(tf.string, shape=(None, conf.input_max_len), name="enc_inps") 
        self.enc_lens = tf.placeholder(tf.int32, shape=[None], name="enc_lens") 

        self.enc_str_topics = tf.placeholder(tf.string, shape=(None, None), name="enc_topics") 

        self.dec_str_inps = tf.placeholder(tf.string, shape=[None, conf.output_max_len + 2], name="dec_inps") 
        self.dec_lens = tf.placeholder(tf.int32, shape=[None], name="dec_lens") 

        # table lookup
        self.enc_inps = self.in_table.lookup(self.enc_str_inps)
        self.enc_topics = self.topic_in_table.lookup(self.enc_str_topics)
        self.dec_inps = self.in_table.lookup(self.dec_str_inps)


        batch_size = tf.shape(self.enc_inps)[0] 

        with variable_scope.variable_scope(self.model_kind, dtype=dtype) as scope: 
            # Create encode graph and get attn states
            graphlg.info("Creating embeddings and do lookup...")
            t_major_enc_inps = tf.transpose(self.enc_inps)
            with ops.device("/cpu:0"):
                self.embedding = variable_scope.get_variable("embedding", [conf.input_vocab_size, conf.embedding_size])
                self.emb_enc_inps = embedding_lookup_unique(self.embedding, t_major_enc_inps)
                self.topic_embedding = variable_scope.get_variable("topic_embedding",
                                                            [conf.topic_vocab_size, conf.topic_embedding_size], trainable=False)
                self.emb_enc_topics = embedding_lookup_unique(self.topic_embedding, self.enc_topics)

            graphlg.info("Creating out projection weights...") 
            if conf.out_layer_size != None:
                w = tf.get_variable("proj_w", [conf.out_layer_size, conf.output_vocab_size], dtype=dtype)
            else:
                w = tf.get_variable("proj_w", [conf.num_units, conf.output_vocab_size], dtype=dtype)
            b = tf.get_variable("proj_b", [conf.output_vocab_size], dtype=dtype)
            self.out_proj = (w, b)

            graphlg.info("Creating encoding dynamic rnn...")
            with variable_scope.variable_scope("encoder", dtype=dtype) as scope: 
                if conf.bidirectional:
                    cell_fw = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
                    cell_bw = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
                    self.enc_outs, self.enc_states = bidirectional_dynamic_rnn(
                                                                cell_fw=cell_fw, cell_bw=cell_bw,
                                                                inputs=self.emb_enc_inps,
                                                                sequence_length=self.enc_lens,
                                                                dtype=dtype,
                                                                parallel_iterations=16,
                                                                time_major=True,
                                                                scope=scope)
                    fw_s, bw_s = self.enc_states 
                    self.enc_states = tuple([tf.concat([f, b], axis=1) for f, b in zip(fw_s, bw_s)])
                    self.enc_outs = tf.concat([self.enc_outs[0], self.enc_outs[1]], axis=2)
                else:
                    cell = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
                    self.enc_outs, self.enc_states = dynamic_rnn(cell=cell,
                                                            inputs=self.emb_enc_inps,
                                                            sequence_length=self.enc_lens,
                                                            parallel_iterations=16,
                                                            scope=scope,
                                                            dtype=dtype,
                                                            time_major=True)
            attn_len = tf.shape(self.enc_outs)[0]
             
            graphlg.info("Preparing init attention and states for decoder...")
            initial_state = self.enc_states
            attn_states = tf.transpose(self.enc_outs, perm=[1, 0, 2])
            attn_size = self.conf.num_units
            topic_attn_size = self.conf.num_units 
            k = tf.get_variable("topic_proj", [1, 1, self.conf.topic_embedding_size, topic_attn_size]) 
            topic_attn_states = nn_ops.conv2d(tf.expand_dims(self.emb_enc_topics, 2), k, [1, 1, 1, 1], "SAME")
            topic_attn_states = tf.squeeze(topic_attn_states, axis=2)

            graphlg.info("Creating decoder cell...")
            with variable_scope.variable_scope("decoder", dtype=dtype) as scope: 
                cell = CreateMultiRNNCell(conf.cell_model, attn_size, conf.num_layers, conf.output_keep_prob)
                # topic
                if not self.for_deploy: 
                    graphlg.info("Embedding decoder inps, tars and tar weights...")
                    t_major_dec_inps = tf.transpose(self.dec_inps)
                    t_major_tars = tf.slice(t_major_dec_inps, [1, 0], [conf.output_max_len + 1, -1])
                    t_major_dec_inps = tf.slice(t_major_dec_inps, [0, 0], [conf.output_max_len + 1, -1])
                    t_major_tar_wgts = tf.cumsum(tf.one_hot(self.dec_lens - 1, conf.output_max_len + 1, axis=0), axis=0, reverse=True)
                    with ops.device("/cpu:0"):
                        emb_dec_inps = embedding_lookup_unique(self.embedding, t_major_dec_inps)

                    hp_train = helper.ScheduledEmbeddingTrainingHelper(inputs=emb_dec_inps, sequence_length=self.enc_lens, 
                                                                       embedding=self.embedding, sampling_probability=0.0,
                                                                       out_proj=self.out_proj, except_ids=None, time_major=True)

                    output_layer = None 
                    my_decoder = AttnTopicDecoder(cell=cell, helper=hp_train, initial_state=initial_state,
                                                    attn_states=attn_states, attn_size=attn_size,
                                                    topic_attn_states=topic_attn_states, topic_attn_size=topic_attn_size,
                                                    output_layer=output_layer)
                    t_major_cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder,
                                                                            output_time_major=True,
                                                                            maximum_iterations=conf.output_max_len + 1,
                                                                            scope=scope)
                    t_major_outs = t_major_cell_outs.rnn_output

                    # Branch 1 for debugging, doesn't have to be called
                    self.outputs = tf.transpose(t_major_outs, perm=[1, 0, 2])
                    L = tf.shape(self.outputs)[1]
                    w, b = self.out_proj
                    self.outputs = tf.reshape(self.outputs, [-1, int(w.shape[0])])
                    self.outputs = tf.matmul(self.outputs, w) + b
                    
                    # For masking the except_ids when debuging
                    #m = tf.shape(self.outputs)[0]
                    #self.mask = tf.zeros([m, int(w.shape[1])])
                    #for i in [3]:
                    #    self.mask = self.mask + tf.one_hot(indices=tf.ones([m], dtype=tf.int32) * i, on_value=100.0, depth=int(w.shape[1]))
                    #self.outputs = self.outputs - self.mask

                    self.outputs = tf.argmax(self.outputs, axis=1)
                    self.outputs = tf.reshape(self.outputs, [-1, L])
                    self.outputs = self.out_table.lookup(tf.cast(self.outputs, tf.int64))

                    # Branch 2 for loss
                    self.loss = dyn_sequence_loss(self.conf, t_major_outs, self.out_proj, t_major_tars, t_major_tar_wgts)
                    self.summary = tf.summary.scalar("%s/loss" % self.name, self.loss)

                    # backpropagation
                    self.build_backprop(self.loss, conf, dtype)

                    #saver
                    self.trainable_params.extend(tf.trainable_variables() + [self.topic_embedding])
                    need_to_save = self.global_params + self.trainable_params + self.optimizer_params + tf.get_default_graph().get_collection("saveable_objects") + [self.topic_embedding] 
                    self.saver = tf.train.Saver(need_to_save, max_to_keep=conf.max_to_keep)
                else:
                    hp_infer = helper.GreedyEmbeddingHelper(embedding=self.embedding,
                                                            start_tokens=tf.ones(shape=[batch_size], dtype=tf.int32),
                                                            end_token=EOS_ID, out_proj=self.out_proj)

                    output_layer = None #layers_core.Dense(self.conf.outproj_from_size, use_bias=True)
                    my_decoder = AttnTopicDecoder(cell=cell, helper=hp_infer, initial_state=initial_state,
                                                    attn_states=attn_states, attn_size=attn_size,
                                                    topic_attn_states=topic_attn_states, topic_attn_size=topic_attn_size,
                                                    output_layer=output_layer)
                    cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, scope=scope, maximum_iterations=40)
                    self.outputs = cell_outs.sample_id
                    #lookup
                    self.outputs = self.out_table.lookup(tf.cast(self.outputs, tf.int64))

                    #saver
                    self.trainable_params.extend(tf.trainable_variables())
                    self.saver = tf.train.Saver(max_to_keep=conf.max_to_keep)

                    # Exporter for serving
                    self.model_exporter = exporter.Exporter(self.saver)
                    inputs = {
                        "enc_inps":self.enc_str_inps,
                        "enc_lens":self.enc_lens
                    } 
                    outputs = {"out":self.outputs}
                    self.model_exporter.init(
                        tf.get_default_graph().as_graph_def(),
                        named_graph_signatures={
                            "inputs": exporter.generic_signature(inputs),
                            "outputs": exporter.generic_signature(outputs)
                        })
                    graphlg.info("Graph done")
                    graphlg.info("")

                self.dec_states = final_state

    def build_backprop(self, loss, conf, dtype):
        # Backprop graph and optimizers
        with variable_scope.variable_scope(self.model_kind, dtype=dtype) as scope: 
            graphlg.info("Creating backpropagation graph and optimizers...")
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
                "RMSProp":tf.train.RMSPropOptimizer(self.learning_rate)
            }

            self.opt = self.optimizers[conf.opt_name]
            tmp = set(tf.global_variables()) 

            if self.job_type == "worker": 
                self.opt = SyncReplicasOptimizer(self.opt, conf.replicas_to_aggregate, conf.total_num_replicas) 
                grads_and_vars = self.opt.compute_gradients(loss=loss) 
                gradients, variables = zip(*grads_and_vars)  
            else:
                gradients = tf.gradients(loss, tf.trainable_variables(), aggregation_method=2)
                variables = tf.trainable_variables()

            clipped_gradients, self.grad_norm = tf.clip_by_global_norm(gradients, conf.max_gradient_norm)
            self.update = self.opt.apply_gradients(zip(clipped_gradients, variables), self.global_step)

            self.optimizer_params.append(self.learning_rate)
            self.optimizer_params.extend(list(set(tf.global_variables()) - tmp))
            self.global_params.extend([self.global_step, self.data_idx])
        return

    def get_init_ops(self, job_type, task_id):
        init_ops = [tf.variables_initializer(set(self.optimizer_params + self.global_params + self.trainable_params))]
        if self.conf.embedding_init:
            w2v = np.load(self.conf.embedding_init)
            init_ops.append(self.embedding.assign(w2v))
            init_ops = list(set(init_ops) - set([self.embedding]))

        if self.conf.topic_embedding_init:
            t2v = np.load(self.conf.topic_embedding_init)
            init_ops.append(self.topic_embedding.assign(t2v))
            init_ops = list(set(init_ops) - set([self.topic_embedding]))

        if not self.for_deploy and task_id == 0:
            vocab_file = filter(lambda x: re.match("vocab[0-9]+\.all", x) != None, os.listdir(self.conf.data_dir))[0]
            f = codecs.open(os.path.join(self.conf.data_dir, vocab_file))
            k = [line.strip() for line in f]
            k = k[0:self.conf.input_vocab_size]
            v = [i for i in range(len(k))]
            op_in = self.in_table.insert(constant_op.constant(k), constant_op.constant(v, dtype=tf.int64))
            op_out = self.out_table.insert(constant_op.constant(v,dtype=tf.int64), constant_op.constant(k))

            # Topic
            topic_vocab_file = "vocab.topic"
            ft = codecs.open(os.path.join(self.conf.data_dir, topic_vocab_file))
            k = [line.strip() for line in ft]
            v = [i for i in range(len(k))]
            op_topic_in = self.topic_in_table.insert(constant_op.constant(k), constant_op.constant(v, dtype=tf.int64))

            init_ops.extend([op_in, op_out, op_topic_in])
        return init_ops

    def get_restorer(self):
        restorer = tf.train.Saver(self.global_params + self.trainable_params + self.optimizer_params +
                                    tf.get_default_graph().get_collection("saveable_objects") + [self.topic_embedding])
        return restorer

    def get_dequeue_op(self):
        return None

    def fetch_test_data(self, records, begin=0, size=128):
        data = []
        for each in records:
            p = each.strip()
            words = tokenize_word(p)
            p_list = words #re.split(" +", p.strip())
            data.append([p_list, len(p_list) + 1, [], 1])
        return data 

    def fetch_train_data(self, records, use_random=True, begin=0, size=128):
        examples = []
        begin = begin % len(records)
        size = len(records) - begin if begin + size >= len(records) else size 
        if use_random == True:
            examples = random.sample(records, size)
        else:
            examples = records[begin:begin+size]

        data = []
        for each in examples:
            p, r, topic = re.split("\t", each.strip())
            p_list = re.split(" +", p.strip())
            r_list = re.split(" +", r.strip())
            topic_list = re.split(" +", topic.strip())
            if self.conf.reverse:
                p_list, r_list = r_list, p_list
            data.append([p_list, len(p_list) + 1, r_list, len(r_list) + 1, topic_list])

        return data 

    def get_batch(self, examples):
        conf = self.conf
        batch_enc_inps, batch_dec_inps, batch_enc_lens, batch_dec_lens, batch_enc_topics = [], [], [], [], []
        for encs, enc_len, decs, dec_len, topic_word in examples:
            # Encoder inputs are padded, reversed and then padded to max.
            enc_len = enc_len if enc_len < conf.input_max_len else conf.input_max_len
            encs = encs[0:conf.input_max_len]
            if conf.enc_reverse:
                encs = list(reversed(encs + ["_PAD"] * (enc_len - len(encs))))
            enc_inps = encs + ["_PAD"] * (conf.input_max_len - len(encs))

            batch_enc_inps.append(enc_inps)
            batch_enc_lens.append(np.int32(enc_len))
            batch_enc_topics.append(topic_word)
            if not self.for_deploy:
                # Decoder inputs with an extra "GO" symbol and "EOS_ID", then padded.
                decs += ["_EOS"]
                decs = decs[0:conf.output_max_len + 1]
                # fit to the max_dec_len
                if dec_len > conf.output_max_len + 1:
                    dec_len = conf.output_max_len + 1

                # Merge dec inps and targets 
                batch_dec_inps.append(["_GO"] + decs + ["_PAD"] * (conf.output_max_len + 1 - len(decs)))
                batch_dec_lens.append(np.int32(dec_len))
        feed_dict = {
                "enc_inps:0": batch_enc_inps,
                "enc_lens:0": batch_enc_lens,
                "dec_inps:0": batch_dec_inps,
                "dec_lens:0": batch_dec_lens,
                "enc_topics:0": batch_enc_topics
        }
        for k, v in feed_dict.items():
            if not v: 
                del feed_dict[k]

        return feed_dict 

    def Project(self, session, records, tensor):
        #embedding = get_dtype=tensor.dtype, tensor_array_name="proj_name", size=len(records), infer_shape=False)
        emb_list = [] 
        out_list = []
        for start in range(0, len(records), self.conf.batch_size):
            batch = records[start:start + self.conf.batch_size]
            examples = self.fetch_test_data(batch, begin=0, size=len(batch))
            input_feed = self.get_batch(examples)
            a = session.run([tensor, self.outputs], feed_dict=input_feed)
            emb_list.append(a[0])
            out_list.append(a[1])
        embs = np.concatenate(emb_list,axis=0)
        outs = np.concatenate(out_list,axis=0)
        return embs, outs
        
         
    def step(self, session, input_feed, forward_only, debug=False, run_options=None, run_metadata=None):

        summary = None
        loss = None
        outputs = None

        if not self.for_deploy:
            output_feed = []
            t = time.time()
            if debug and forward_only:
                output_feed = [self.summary, self.loss, self.outputs]
                outs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
                summary, loss, outputs = outs[0], outs[1], outs[2]

            elif debug and not forward_only:
                output_feed = [self.summary, self.loss, self.outputs, self.update]
                outs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
                summary, loss, outputs = outs[0], outs[1], outs[2]

            elif not debug and forward_only:
                output_feed = [self.summary, self.loss]
                outs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
                summary, loss = outs[0], outs[1]

            else:
                output_feed = [self.summary, self.loss, self.update]
                outs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
                summary, loss = outs[0], outs[1]
            graphlg.info("TIME %.3f" % (time.time() - t))

        else:
            output_feed = [self.outputs]
            outputs, = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
        return summary, loss, outputs


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(allow_growth=True, allocator_type="BFC")
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=gpu_options,
                                        intra_op_parallelism_threads=16)
    name = "newdyn-topic"

    if len(sys.argv) == 1: 
        with tf.device("/gpu:7"):
            model = DynAttnTopicSeq2Seq(name=name, for_deploy=False)
            model.build()

        init_ops = model.get_init_ops(job_type="single", task_id=0)

        path = os.path.join(confs[name].data_dir, "train.data")
        qr = QueueReader(filename_list=[path])
        deq_batch_record = qr.batched(batch_size=512, min_after_dequeue=2)

        
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
            t0 = time.time()
            output = model.step(sess, feed_dict, False)
            t = time.time() - t0
            print "TIME: %.4f, LOSS: %.10f" % (t, output[1])

    elif sys.argv[1] == "test":
        with tf.device("/gpu:7"):
            model = DynAttnTopicSeq2Seq(name=name, for_deploy=True)
            model.build()

        ops = model.get_init_ops("single", 0)
        print ops
        sess = tf.Session(config=session_config)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(ops[1])

        restorer = model.get_restorer()
        ckpt = tf.train.get_checkpoint_state("../runtime/%s" % name, latest_filename=None)
        restorer.restore(save_path=ckpt.model_checkpoint_path, sess=sess)
        while True:
            query = raw_input("query>>")
            topics = raw_input("topic>>") 
            words = tokenize_word(query)
            examples = [(words, len(words) + 4, [], 1, topics.split())]
            input_feed = model.get_batch(examples)
            outputs = model.step(session=sess, input_feed=input_feed, forward_only=True)
            print "".join(outputs[2][0])

    
