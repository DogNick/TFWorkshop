#from __future__ import absolute_import
import sys
#sys.path.insert(0, "/search/odin/Nick/_python_build2")
import re
import time
import random
import numpy as np
import tensorflow as tf
from ModelCore import *
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.contrib import lookup
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

from tensorflow.python.training.sync_replicas_optimizer import SyncReplicasOptimizer
from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique
from tensorflow.contrib.rnn import  MultiRNNCell, AttentionCellWrapper, GRUCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.seq2seq.python.ops import loss

from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.util import nest

import Nick_plan
import logging as log

graphlg = log.getLogger("graph")
DynamicAttentionWrapper = dynamic_attention_wrapper.DynamicAttentionWrapper
DynamicAttentionWrapperState = dynamic_attention_wrapper.DynamicAttentionWrapperState 
Bahdanau = dynamic_attention_wrapper.BahdanauAttention
Luong = dynamic_attention_wrapper.LuongAttention

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

        #single_cell = DeviceWrapper(ResidualWrapper(single_cell), device='/gpu:%d' % i)
        #single_cell = DeviceWrapper(single_cell, device='/gpu:%d' % i)

        cells.append(single_cell)
    return MultiRNNCell(cells) 

class AttnSeq2Seq(ModelCore):
    def __init__(self, name, job_type="single", task_id=0, dtype=tf.float32):
        super(AttnSeq2Seq, self).__init__(name, job_type, task_id, dtype) 
        self.embedding = None
        self.out_proj = None

    def build(self, for_deploy, variants=""):
        conf = self.conf
        name = self.name
        job_type = self.job_type
        dtype = self.dtype
        self.beam_size = 1 if (not for_deploy or variants=="score") else sum(self.conf.beam_splits)

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
        self.enc_str_inps = tf.placeholder(tf.string, shape=(None, conf.input_max_len), name="enc_inps") 
        self.enc_lens = tf.placeholder(tf.int32, shape=[None], name="enc_lens") 
        self.dec_str_inps = tf.placeholder(tf.string, shape=[None, conf.output_max_len + 2], name="dec_inps") 
        self.dec_lens = tf.placeholder(tf.int32, shape=[None], name="dec_lens") 
        self.down_wgts = tf.placeholder(tf.float32, shape=[None], name="down_wgts")

        # lookup
        self.enc_inps = self.in_table.lookup(self.enc_str_inps)
        self.dec_inps = self.in_table.lookup(self.dec_str_inps)


        with variable_scope.variable_scope(self.model_kind, dtype=dtype) as scope: 
            # Create encode graph and get attn states
            graphlg.info("Creating embeddings and embedding enc_inps.")
            with ops.device("/cpu:0"):
                self.embedding = variable_scope.get_variable("embedding", [conf.output_vocab_size, conf.embedding_size])
                self.emb_enc_inps = embedding_lookup_unique(self.embedding, self.enc_inps)

            graphlg.info("Creating dynamic rnn...")
            if conf.bidirectional:
                with variable_scope.variable_scope("encoder", dtype=dtype) as scope: 
                    cell_fw = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
                    cell_bw = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
                self.enc_outs, self.enc_states = bidirectional_dynamic_rnn(
                                                                cell_fw=cell_fw, cell_bw=cell_bw,
                                                                inputs=self.emb_enc_inps,
                                                                sequence_length=self.enc_lens,
                                                                dtype=dtype,
                                                                parallel_iterations=16,
                                                                scope=scope)

                fw_s, bw_s = self.enc_states 
                self.enc_states = []
                for f, b in zip(fw_s, bw_s):
                    if isinstance(f, LSTMStateTuple):
                        self.enc_states.append(LSTMStateTuple(tf.concat([f.c, b.c], axis=1), tf.concat([f.h, b.h], axis=1)))
                    else:
                        self.enc_states.append(tf.concat([f, b], 1))
                self.enc_outs = tf.concat([self.enc_outs[0], self.enc_outs[1]], axis=2)
                mem_size = 2 * conf.num_units
                enc_state_size = 2 * conf.num_units 
            else:
                with variable_scope.variable_scope("encoder", dtype=dtype) as scope: 
                    cell = CreateMultiRNNCell(conf.cell_model, conf.num_units, conf.num_layers, conf.output_keep_prob)
                self.enc_outs, self.enc_states = dynamic_rnn(cell=cell,
                                                            inputs=self.emb_enc_inps,
                                                            sequence_length=self.enc_lens,
                                                            parallel_iterations=16,
                                                            scope=scope,
                                                            dtype=dtype)
                mem_size = conf.num_units
                enc_state_size = conf.num_units

            memory = tf.reshape(tf.concat([self.enc_outs] * self.beam_size, 2), [-1, conf.input_max_len, mem_size])
            memory_lens = tf.squeeze(tf.reshape(tf.concat([tf.expand_dims(self.enc_lens, 1)] * self.beam_size, 1), [-1, 1]), 1)
            batch_size = tf.shape(self.enc_outs)[0]

            graphlg.info("Creating out_proj...") 
            if conf.out_layer_size:
                w = tf.get_variable("proj_w", [conf.out_layer_size, conf.output_vocab_size], dtype=dtype)
            else:
                w = tf.get_variable("proj_w", [mem_size, conf.output_vocab_size], dtype=dtype)
            b = tf.get_variable("proj_b", [conf.output_vocab_size], dtype=dtype)
            self.out_proj = (w, b)

            graphlg.info("Preparing decoder inps...")
            dec_inps = tf.slice(self.dec_inps, [0, 0], [-1, conf.output_max_len + 1])
            with ops.device("/cpu:0"):
                emb_dec_inps = embedding_lookup_unique(self.embedding, dec_inps)


            # Attention  
            with variable_scope.variable_scope("decoder", dtype=dtype) as scope: 
                decoder_cell = CreateMultiRNNCell(conf.cell_model, enc_state_size, conf.num_layers, conf.output_keep_prob)
            max_mem_size = self.conf.input_max_len + self.conf.output_max_len + 2
            if conf.attention == "Luo":
                mechanism = dynamic_attention_wrapper.LuongAttention(num_units=mem_size, memory=memory, max_mem_size=max_mem_size,
                                                                        memory_sequence_length=memory_lens)
            elif conf.attention == "Bah":
                mechanism = dynamic_attention_wrapper.BahdanauAttention(num_units=mem_size, memory=memory, max_mem_size=max_mem_size,
                                                                        memory_sequence_length=memory_lens)
            else:
                print "Unknown attention stype, must be Luo or Bah" 
                exit(0)

            attn_cell = DynamicAttentionWrapper(cell=decoder_cell, attention_mechanism=mechanism,
                                                attention_size=mem_size, addmem=self.conf.addmem)

            # Zeros for initial state
            zero_attn_states = attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size * self.beam_size)
            

            # Encoder states for initial state 
            init_states = []
            for i, each in enumerate(self.enc_states):
              if isinstance(each, LSTMStateTuple):
                new_c = tf.reshape(tf.concat([each.c] * self.beam_size, 1), [-1, enc_state_size], name="init_lstm_c")
                new_h = tf.reshape(tf.concat([each.h] * self.beam_size, 1), [-1, enc_state_size], name="init_lstm_h")
                init_states.append(LSTMStateTuple(new_c, new_h))
              else:
                init_states.append(tf.reshape(tf.concat([each] * self.beam_size, 1), [-1, enc_state_size], name="init_cell_state"))

            zero_attn_states = DynamicAttentionWrapperState(tuple(init_states), zero_attn_states.attention, zero_attn_states.newmem, zero_attn_states.alignments)

            if not for_deploy: 
                dec_init_state = zero_attn_states
                hp_train = helper.ScheduledEmbeddingTrainingHelper(inputs=emb_dec_inps, sequence_length=self.dec_lens, 
                                                                   embedding=self.embedding, sampling_probability=0.0,
                                                                   out_proj=self.out_proj)
                output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
                my_decoder = basic_decoder.BasicDecoder(cell=attn_cell, helper=hp_train, initial_state=dec_init_state, output_layer=output_layer)
                cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, impute_finished=(self.conf.addmem==False),
                                                                maximum_iterations=conf.output_max_len + 1, scope=scope)
                outputs = cell_outs.rnn_output

                L = tf.shape(outputs)[1]
                outputs = tf.reshape(outputs, [-1, int(self.out_proj[0].shape[0])])
                outputs = tf.matmul(outputs, self.out_proj[0]) + self.out_proj[1] 
                logits = tf.reshape(outputs, [-1, L, int(self.out_proj[0].shape[1])])

                # branch 1 for debugging, doesn't have to be called
                #m = tf.shape(self.outputs)[0]
                #self.mask = tf.zeros([m, int(w.shape[1])])
                #for i in [3]:
                #    self.mask = self.mask + tf.one_hot(indices=tf.ones([m], dtype=tf.int32) * i, on_value=100.0, depth=int(w.shape[1]))
                #self.outputs = self.outputs - self.mask

                self.outputs = tf.argmax(logits, axis=2)
                self.outputs = tf.reshape(self.outputs, [-1, L])
                self.outputs = self.out_table.lookup(tf.cast(self.outputs, tf.int64))

                # branch 2 for loss
                tars = tf.slice(self.dec_inps, [0, 1], [-1, L])
                wgts = tf.cumsum(tf.one_hot(self.dec_lens, L), axis=1, reverse=True)

                # average over batch and length also with downsaple first n-gram

                #wgts = tf.cumsum(tf.where(3 < L, tf.one_hot(tf.ones([batch_size], tf.int32) * 3, L), tf.ones([batch_size, L])), axis=1, reverse=True, exclusive=True) * tf.expand_dims(self.down_wgts, 1)
                #self.loss = loss.sequence_loss(logits=logits, targets=tars, weights=wgts, average_across_timesteps=True, average_across_batch=True)

                # average over batch but not over length
                wgts = wgts * tf.expand_dims(self.down_wgts, 1)
                self.loss = loss.sequence_loss(logits=logits, targets=tars, weights=wgts, average_across_timesteps=False, average_across_batch=False)
                self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.down_wgts)
                see_loss = self.loss

                tf.summary.scalar("loss", see_loss)
                self.summary_ops = tf.summary.merge_all()
                self.update = self.backprop(self.loss)
                self.train_outputs_map = {
                        "loss":see_loss,
                        "update":self.update
                }
                self.fo_outputs_map = {
                        "loss":see_loss
                }
                self.debug_outputs_map = {
                        "loss":see_loss,
                        "outputs":self.outputs,
                        "update":self.update
                }

                #saver
                self.trainable_params.extend(tf.trainable_variables())
                self.saver = tf.train.Saver(max_to_keep=conf.max_to_keep)

            else:
                if variants == "score":
                    dec_init_state = zero_attn_states
                    hp_train = helper.ScheduledEmbeddingTrainingHelper(inputs=emb_dec_inps, sequence_length=self.dec_lens, 
                                                                       embedding=self.embedding, sampling_probability=0.0,
                                                                       out_proj=self.out_proj)
                    output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
                    my_decoder = score_decoder.ScoreDecoder(cell=attn_cell, helper=hp_train, out_proj=self.out_proj, initial_state=dec_init_state, output_layer=output_layer)
                    cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, scope=scope, maximum_iterations=self.conf.output_max_len, impute_finished=False)
                    L = tf.shape(cell_outs.logprobs)[1]
                    one_hot = tf.one_hot(tf.slice(self.dec_inps, [0, 1], [-1, L]), depth=self.conf.output_vocab_size, axis=-1, on_value=1.0, off_value=0.0)
                    outputs = tf.reduce_sum(cell_outs.logprobs * one_hot, 2)
                    #outputs = tf.Print(outputs, [tf.shape(outputs)], message="outputs", summarize=1000)
                    outputs = tf.reduce_sum(outputs, axis=1)
                    self.infer_outputs_map["logprobs"] = outputs
                else:
                    dec_init_state = beam_decoder.BeamState(tf.zeros([batch_size * self.beam_size]), zero_attn_states, tf.zeros([batch_size * self.beam_size], tf.int32))
                    #dec_init_state = nest.map_structure(lambda x:tf.Print(x, [tf.shape(x)], message=str(x)+"dec_init"), dec_init_state)

                    hp_infer = helper.GreedyEmbeddingHelper(embedding=self.embedding,
                                                            start_tokens=tf.ones(shape=[batch_size * self.beam_size], dtype=tf.int32),
                                                            end_token=EOS_ID, out_proj=self.out_proj)

                    output_layer = layers_core.Dense(self.conf.out_layer_size, use_bias=True) if self.conf.out_layer_size else None
                    my_decoder = beam_decoder.BeamDecoder(cell=attn_cell, helper=hp_infer, out_proj=self.out_proj, initial_state=dec_init_state,
                                                            beam_splits=self.conf.beam_splits, max_res_num=self.conf.max_res_num, output_layer=output_layer)
                    cell_outs, final_state = decoder.dynamic_decode(decoder=my_decoder, scope=scope, maximum_iterations=self.conf.output_max_len, impute_finished=True)

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

                    ## Creating tail_ids 
                    #batch_size = tf.Print(batch_size, [batch_size], message="AttnSeq2Seq batch")
                    #beam_symbols = tf.Print(cell_outs.beam_symbols, [tf.shape(cell_outs.beam_symbols)], message="beam_symbols")
                    #beam_parents = tf.Print(cell_outs.beam_parents, [tf.shape(cell_outs.beam_parents)], message="beam_parents")
                    #beam_ends = tf.Print(cell_outs.beam_ends, [tf.shape(cell_outs.beam_ends)], message="beam_ends") 
                    #beam_end_parents = tf.Print(cell_outs.beam_end_parents, [tf.shape(cell_outs.beam_end_parents)], message="beam_end_parents") 
                    #beam_end_probs = tf.Print(cell_outs.beam_end_probs, [tf.shape(cell_outs.beam_end_probs)], message="beam_end_probs") 
                    #alignments = tf.Print(cell_outs.alignments, [tf.shape(cell_outs.alignments)], message="beam_attns")

                    batch_offset = tf.expand_dims(tf.cumsum(tf.ones([batch_size, self.beam_size], dtype=tf.int32) * self.beam_size, axis=0, exclusive=True), 2)
                    offset2 = tf.expand_dims(tf.cumsum(tf.ones([batch_size, self.beam_size * 2], dtype=tf.int32) * self.beam_size, axis=0, exclusive=True), 2)

                    out_len = tf.shape(beam_symbols)[1]
                    self.beam_symbol_strs = tf.reshape(self.out_table.lookup(tf.cast(beam_symbols, tf.int64)), [batch_size, self.beam_size, -1])
                    self.beam_parents = tf.reshape(beam_parents, [batch_size, self.beam_size, -1]) - batch_offset

                    self.beam_ends = tf.reshape(beam_ends, [batch_size, self.beam_size * 2, -1])
                    self.beam_end_parents = tf.reshape(beam_end_parents, [batch_size, self.beam_size * 2, -1]) - offset2
                    self.beam_end_probs = tf.reshape(beam_end_probs, [batch_size, self.beam_size * 2, -1])
                    self.beam_attns = tf.reshape(alignments, [batch_size, self.beam_size, out_len, -1])

                    self.infer_outputs_map["beam_symbols"] = self.beam_symbol_strs
                    self.infer_outputs_map["beam_parents"] = self.beam_parents
                    self.infer_outputs_map["beam_ends"] = self.beam_ends
                    self.infer_outputs_map["beam_end_parents"] = self.beam_end_parents
                    self.infer_outputs_map["beam_end_probs"] = self.beam_end_probs
                    self.infer_outputs_map["beam_attns"] = self.beam_attns

                    #cell_outs.alignments
                    #self.outputs = tf.concat([outputs_str, tf.cast(cell_outs.beam_parents, tf.string)], 1)

                    #ones = tf.ones([batch_size, self.beam_size], dtype=tf.int32)
                    #aux_matrix = tf.cumsum(ones * self.beam_size, axis=0, exclusive=True)
                    #tail_ids = tf.reshape(tf.cumsum(ones, axis=1, exclusive=True) + aux_matrix, [-1])
                    ##tail_ids = tf.Print(tail_ids, [tf.shape(tail_ids)], message="tail_ids")

                    #tm_beam_parents_reverse = tf.reverse(tf.transpose(cell_outs.beam_parents), axis=[0])
                    #beam_probs = final_state[1] 

                    #def traceback(prev_out, curr_input):
                    #    return tf.gather(curr_input, prev_out) 
                    #    
                    ##tail_ids = tf.Print(tail_ids, [tf.shape(tail_ids)], message="tail_ids")
                    #tm_symbol_index_reverse = tf.scan(traceback, tm_beam_parents_reverse, initializer=tail_ids)

                    ##tm_symbol_index_reverse = tf.Print(tm_symbol_index_reverse, [tf.shape(tm_symbol_index_reverse)], message="tm_symbol_index_reverse")

                    ## Create beam index for symbols, and other info  
                    #tm_symbol_index = tf.concat([tf.expand_dims(tail_ids, 0), tm_symbol_index_reverse], axis=0)

                    ##tm_symbol_index = tf.Print(tm_symbol_index, [tf.shape(tm_symbol_index)], message="tm_symbol_index_1")

                    #tm_symbol_index = tf.reverse(tm_symbol_index, axis=[0])
                    #tm_symbol_index = tf.slice(tm_symbol_index, [1, 0], [-1, -1])

                    ##tm_symbol_index = tf.Print(tm_symbol_index, [tf.shape(tm_symbol_index)], message="tm_symbol_index")

                    #symbol_index = tf.expand_dims(tf.transpose(tm_symbol_index), axis=2)
                    #symbol_index = tf.concat([symbol_index, tf.cumsum(tf.ones_like(symbol_index), exclusive=True, axis=1)], axis=2)

                    ##symbol_index = tf.Print(symbol_index, [tf.shape(symbol_index)], message="symbol_index")
                    ## index alignments and output symbols
                    #alignments = tf.gather_nd(cell_outs.alignments, symbol_index)
                    #symbol_ids = tf.gather_nd(cell_outs.beam_symbols, symbol_index)

                    # outputs and other info
                    #symbol_ids = tf.reshape(symbol_ids, [1, -1])
                    #symbol_ids = tf.Print(symbol_ids, [tf.shape(symbol_ids)], message="symbol_shape")
                    #self.others = [alignments, beam_probs]
                    #self.outputs = self.out_table.lookup(tf.cast(symbol_ids, tf.int64))

                #saver
                self.trainable_params.extend(tf.trainable_variables())
                self.saver = tf.train.Saver(max_to_keep=conf.max_to_keep)

                # Exporter for serving
                self.model_exporter = exporter.Exporter(self.saver)
                inputs = {
                    "enc_inps:0":self.enc_str_inps,
                    "enc_lens:0":self.enc_lens
                } 
                outputs = self.infer_outputs_map
                self.model_exporter.init(
                    tf.get_default_graph().as_graph_def(),
                    named_graph_signatures={
                        "inputs": exporter.generic_signature(inputs),
                        "outputs": exporter.generic_signature(outputs)
                    })
                graphlg.info("Graph done")
                graphlg.info("")

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

	def after_proc(self, out):
		outputs, probs, attns = Nick_plan.handle_beam_out(out, self.conf.beam_splits)
		after_proc_out = {
			"outputs":outputs,
			"probs":probs,
			"attns":attns
		}
		return after_proc_out 


if __name__ == "__main__":
    #name = "attn-s2s-srt-reddit-voca-filtered"
    #name = "attn-s2s-srt-reddit-proced3"
    #name = "attn-s2s-srt-reddit-0init_bah"
    #name = "attn-s2s-stc-all-clean"
    #name = "attn-s2s-srt-reddit-small"
    #name = "attn-s2s-merge-stc-weibo-downsample"
    #name = "attn-s2s-all-downsample-addmem"
    #name = "attn-bi-s2s-all-downsample-addmem"
    #name = "attn-bi-s2s-all-downsample-addmem2"
    name = "attn-s2s-all-downsample-n-gram-addmem"

    #name = "attn-bi-s2s-addmem-poem"
    #name = "attn-bi-s2s-addmem-poem2"
    #name = "attn-bi-poem"
    name = "attn-bi-poem-no-ave-len"
    model = AttnSeq2Seq(name)
    if len(sys.argv) == 2:
        gpu = 0
    flag = sys.argv[1]
    model(flag, use_seg=False)
    #model(flag, use_seg=True)
