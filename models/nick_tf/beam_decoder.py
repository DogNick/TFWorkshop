# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A class of Decoders that may sample to generate the next input.
"""

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
#sys.path.insert(0, "/search/odin/Nick/_python_build2")
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import rnn_cell_impl
import decoder
#from tensorflow.contrib.seq2seq.python.ops import helper as helper_py, dynamic_attention_wrapper 
import helper as helper_py
import dynamic_attention_wrapper 
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

LSTMStateTuple = rnn_cell_impl.LSTMStateTuple
DynamicAttentionWrapperState = dynamic_attention_wrapper.DynamicAttentionWrapperState
DynamicAttentionWrapper = dynamic_attention_wrapper.DynamicAttentionWrapper
EOS_ID=2


__all__ = [
	"BeamDecoderOutput",
	"BeamState",
	"BeamDecoder",
]


#class BasicDecoderOutput(
#	collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
#  pass

class BeamDecoderOutput(
		collections.namedtuple("BeamDecoderOutput", ("rnn_output", "beam_symbols", "beam_parents", "beam_ends", "beam_end_parents", "beam_end_probs", "alignments"))):
  pass

class BeamState(
		collections.namedtuple("BeamState", ("beam_probs", "beam_cell_states", "beam_res_num"))):
  pass

class BeamDecoder(decoder.Decoder):
  def __init__(self, cell, helper, initial_state, out_proj, beam_splits=[1], max_res_num=10,  output_layer=None):
	if not isinstance(cell, rnn_cell_impl.RNNCell):
	  raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
	#if not isinstance(helper, helper_py.Helper):
	#  raise TypeError("helper must be a Helper, received: %s" % type(helper))
	if (output_layer is not None
		and not isinstance(output_layer, layers_base.Layer)):  # pylint: disable=protected-access
	  raise TypeError(
		  "output_layer must be a Layer, received: %s" % type(output_layer))
	self._cell = cell
	self._helper = helper
	self._initial_state = initial_state
	self._output_layer = output_layer
	self._beam_size = sum(beam_splits)
	self._beam_splits = beam_splits 
	self._out_proj = out_proj
	self._max_res_num = max_res_num

  @property
  def batch_size(self):
	batch_size = self._helper.batch_size
	return batch_size 

  @property
  def state_shape(self):
	  return BeamState(beam_probs=tf.TensorShape([None]), beam_cell_states=self._cell.state_shape, beam_res_num=tf.TensorShape([None]))

  @property
  def output_shape(self):
	  return BeamDecoderOutput(
		rnn_output=tensor_shape.TensorShape([]),
		beam_symbols=tensor_shape.TensorShape([]),
		beam_parents=tensor_shape.TensorShape([]),
		beam_ends=tensor_shape.TensorShape([]),
		beam_end_parents=tensor_shape.TensorShape([]),
		beam_end_probs=tensor_shape.TensorShape([]),
		#alignments=tensor_shape.TensorShape([None, self._cell._attention_mechanism._max_mem_size]))
		alignments=tensor_shape.TensorShape([]))

  @property
  def zero_outputs(self):
	  out_size = self.output_size
	  out_type = self.output_dtype
	  def _t(s):
		return (s if isinstance(s, ops.Tensor) else constant_op.constant(
		    tensor_shape.TensorShape(s).as_list(),
		    dtype=dtypes.int32,
		    name="zero_suffix_shape"))
	  beam_ends = tf.zeros(tf.concat([[self.batch_size], _t(out_size.beam_ends)], 0), out_type.beam_ends, name="be")
	  beam_ends = tf.Print(beam_ends, [tf.shape(beam_ends)], message="beam_ends")

	  return BeamDecoderOutput(
		rnn_output=tf.zeros(tf.concat([[self.batch_size],_t(out_size.rnn_output)], 0), out_type.rnn_output),
		beam_symbols=tf.zeros(tf.concat([[self.batch_size],  _t(out_size.beam_symbols)], 0), out_type.beam_symbols, name="bs"),
		beam_parents=tf.zeros(tf.concat([[self.batch_size],  _t(out_size.beam_parents)], 0), out_type.beam_parents, name="bp"),
		beam_ends=beam_ends,
		beam_end_parents=tf.zeros(tf.concat([[self.batch_size], _t(out_size.beam_end_parents)], 0), out_type.beam_end_parents, name="bep"),
		beam_end_probs=tf.zeros(tf.concat([[self.batch_size], _t(out_size.beam_end_probs)], 0), out_type.beam_end_probs, name="bepr"),
		alignments=tf.zeros(tf.concat([[self.batch_size], _t(out_size.alignments)], 0), out_type.alignments))


  def _rnn_output_size(self):
	size = self._cell.output_size
	if self._output_layer is None:
	  return size
	else:
	  # To use layer's compute_output_shape, we need to convert the
	  # RNNCell's output_size entries into shapes with an unknown
	  # batch size.  We then pass this through the layer's
	  # compute_output_shape and read off all but the first (batch)
	  # dimensions to get the output size of the rnn with the layer
	  # applied to the top.
	  output_shape_with_unknown_batch = nest.map_structure(
		  lambda s: tensor_shape.TensorShape([None]).concatenate(s),
		  size)
	  layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
		  output_shape_with_unknown_batch)
	  return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
	# Return the cell output and the id
	if isinstance(self._cell, DynamicAttentionWrapper): 
		alignments_shape = [1, self._cell._attention_mechanism._max_mem_size] 
	else:
		alignments_shape = [] 
	return BeamDecoderOutput(
		rnn_output=self._rnn_output_size(),
		beam_symbols=tensor_shape.TensorShape([]),
		beam_parents=tensor_shape.TensorShape([]),
		beam_ends=tensor_shape.TensorShape([2]),
		beam_end_parents=tensor_shape.TensorShape([2]),
		beam_end_probs=tensor_shape.TensorShape([2]),
		alignments=tensor_shape.TensorShape(alignments_shape))

  @property
  def output_dtype(self):
	# Assume the dtype of the cell is the output_size structure
	# containing the input_state's first component's dtype.
	# Return that structure and int32 (the id)
	dtype = nest.flatten(self._initial_state)[0].dtype
	return BeamDecoderOutput(
		nest.map_structure(lambda _: dtype, self._rnn_output_size()),
		dtypes.int32,
		dtypes.int32,
		dtypes.int32,
		dtypes.int32,
		dtypes.float32,
		dtypes.float32)

  def initialize(self, name=None):
	"""Initialize the decoder.

	Args:
	  name: Name scope for any created operations.

	Returns:
	  `(finished, first_inputs, initial_state)`.
	"""

	#self._initial_state = nest.map_structure(lambda x: tf.Print(x, [tf.shape(x)], message=str(x)+"_init"), self._initial_state)
	return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
	"""Perform a decoding step.

	Args:
	  time: scalar `int32` tensor.
	  inputs: A (structure of) input tensors.
	  state: A (structure of) state tensors and TensorArrays.
	  name: Name scope for any created operations.

	Returns:
	  `(outputs, next_state, next_inputs, finished)`.
	"""
	with ops.name_scope(name, "BeamDecoderStep", (time, inputs, state)):

	  beam_probs, attn_state, res_num = state.beam_probs, state.beam_cell_states, state.beam_res_num
	  
	  cell_outputs, cell_state = self._cell(inputs, attn_state)
	  if self._output_layer is not None:
		cell_outputs = self._output_layer(cell_outputs)

	  w, b = self._out_proj 
	  beam_logits = tf.nn.xw_plus_b(cell_outputs, w, b) 

	  probs = tf.expand_dims(beam_probs, 1) + tf.nn.log_softmax(beam_logits)
	  batch_beam_probs = tf.reshape(probs, [-1, self._beam_size * int(w.shape[1])])
	  def f1():
		first_beam_probs = tf.slice(batch_beam_probs, [0, 0], [-1, int(w.shape[1])])
		values, indices = tf.nn.top_k(first_beam_probs, k=self._beam_size)

		beam_probs = tf.reshape(values, [-1])
		offset = tf.reshape(tf.ones([self.batch_size], dtype=tf.int32) * self._beam_size, [-1, self._beam_size])
		beam_offset = tf.cumsum(offset, axis=0, exclusive=True)

		beam_symbols = tf.reshape(indices % int(w.shape[1]), [-1])
		beam_parents = tf.reshape(indices // int(w.shape[1]) + beam_offset, [-1])
		beam_ends = tf.zeros([self.batch_size * 2], tf.int32)
		beam_end_parents = tf.zeros([self.batch_size * 2], tf.int32)
		beam_end_probs = tf.zeros([self.batch_size * 2])
		new_res_cnt = tf.zeros([tf.cast(self.batch_size / self._beam_size, tf.int32)], tf.int32)
		#beam_end_parents = tf.Print(beam_end_parents, [tf.shape(beam_end_parents)], message="f1_bep")
		return beam_probs, beam_parents, beam_symbols, beam_ends, beam_end_parents, beam_end_probs, new_res_cnt 

	  def f2():
		n = int(w.shape[1])
		sp_size = [b * n for b in self._beam_splits]
		batch_beam_probs_list = tf.split(batch_beam_probs, sp_size, axis = 1)
		offset = tf.reshape(tf.ones([self.batch_size], dtype=tf.int32) * self._beam_size, [-1, self._beam_size])
		beam_offsets = tf.cumsum(offset, axis=0, exclusive=True)
		beam2_offsets = tf.concat([beam_offsets] * 2, 1) 

		multi_beam_probs = []
		multi_beam_parents = []
		multi_beam_symbols = []
		multi_beam_ends = []
		multi_beam_end_parents = []
		multi_beam_end_probs = []

		for i, bs in enumerate(self._beam_splits): 
		    prob_2b, idx_2b = tf.nn.top_k(batch_beam_probs_list[i], k=2*bs)
		    in_beam_offset = sum(self._beam_splits[0:i]) 
		    seg_2b_symbols = idx_2b % int(w.shape[1])
		    seg_2b_parents = idx_2b // int(w.shape[1]) + in_beam_offset
		    seg_2b_ends = tf.where(tf.equal(seg_2b_symbols, EOS_ID),
		                  tf.ones(tf.shape(seg_2b_symbols), tf.int32),
		                  tf.zeros(tf.shape(seg_2b_symbols), tf.int32))
		    multi_beam_end_probs.append(prob_2b)
		    multi_beam_ends.append(seg_2b_ends)
		    multi_beam_end_parents.append(seg_2b_parents)

		    prob, idx = tf.nn.top_k(prob_2b + tf.cast(seg_2b_ends * -100000, tf.float32), k=bs)  
		    idx = tf.expand_dims(idx, 2)
		    idx = tf.concat([tf.cumsum(tf.ones_like(idx), exclusive=True, axis=0), idx], axis=2)
		    seg_b_symbols = tf.gather_nd(seg_2b_symbols, idx) 
		    seg_b_parents = tf.gather_nd(seg_2b_parents, idx) 

		    multi_beam_probs.append(prob)
		    multi_beam_symbols.append(seg_b_symbols)
		    multi_beam_parents.append(seg_b_parents)

		multi_beam_probs = tf.reshape(tf.concat(multi_beam_probs, 1), [-1])
		multi_beam_parents = tf.reshape(tf.concat(multi_beam_parents, 1) + beam_offsets, [-1])
		multi_beam_symbols = tf.reshape(tf.concat(multi_beam_symbols, 1), [-1])
		ends = tf.concat(multi_beam_ends, 1)
		multi_beam_ends = tf.reshape(ends, [-1])
		multi_beam_end_parents = tf.reshape(tf.concat(multi_beam_end_parents, 1) + beam2_offsets, [-1])
		multi_beam_end_probs = tf.reshape(tf.concat(multi_beam_end_probs, 1), [-1])
		new_res_cnt = tf.reduce_sum(ends, 1)

		return multi_beam_probs, multi_beam_parents, multi_beam_symbols, multi_beam_ends, multi_beam_end_parents, multi_beam_end_probs, new_res_cnt

	  beam_probs, beam_parents, beam_symbols, beam_ends, beam_end_parents, beam_end_probs, new_res_cnt = tf.cond(tf.equal(time, 0), f1, f2)

	  # Reorder for next 
	  # TODO problems here about newmem, should be reindexed ?
	  next_states = []
	  if isinstance(self._cell, DynamicAttentionWrapper): 
		  states = cell_state.cell_state
	  else:
		  states = cell_state

	  with ops.device("/cpu:0"):
		for each in states:
		  if isinstance(each, LSTMStateTuple):
		    next_states.append(LSTMStateTuple(tf.nn.embedding_lookup(each.c, beam_parents), tf.nn.embedding_lookup(each.h, beam_parents)))
		  else:
		    next_states.append(tf.nn.embedding_lookup(each, beam_parents))


	  if isinstance(self._cell, DynamicAttentionWrapper):
		with ops.device("/cpu:0"):
		  next_attns = tf.nn.embedding_lookup(cell_state.attention, beam_parents)
		  alignments = tf.nn.embedding_lookup(cell_state.alignments, beam_parents)
		  
		if self._cell._addmem:
		  with ops.device("/cpu:0"):
		    reordered_outputs = tf.nn.embedding_lookup(cell_outputs, beam_parents)
		    newmem = cell_state.newmem.write(time + 1, reordered_outputs)
		else:
		    newmem = cell_state.newmem
		beam_states = DynamicAttentionWrapperState(tuple(next_states), next_attns, newmem, alignments)

	  else:
		alignments = tf.zeros([self.batch_size])  
		beam_states = tuple(next_states)

	  (finished, next_inputs, next_state) = self._helper.next_inputs(
		  time=time,
		  outputs=cell_outputs,
		  state=beam_states,
		  sample_ids=beam_symbols)

	  #next_state = nest.map_structure(lambda x: tf.Print(x, [tf.shape(x)], message=str(x)+"_cell_state"), next_state) 
	  
	  #res_num = tf.Print(res_num, [tf.shape(res_num)], message="res_num", summarize=100000)
	  max_num = tf.ones_like(res_num) * self._max_res_num
	  #new_res_cnt = tf.Print(new_res_cnt, [tf.shape(new_res_cnt)], message="new_res_cnt")
	  res_num = tf.slice(tf.reshape(res_num, [-1, self._beam_size]), [0, 0], [-1, 1])
	  #res_num = tf.Print(res_num, [tf.shape(res_num)], message="res_num2", summarize=100000)
	  batch_res_num = res_num + tf.expand_dims(new_res_cnt, 1)

	  #batch_res_num = tf.Print(batch_res_num, [tf.shape(batch_res_num)], message="batch_res_num", summarize=100000)
	  new_res_num = tf.reshape(tf.concat([batch_res_num] * self._beam_size, 1), [-1])

	  #new_res_num = tf.Print(new_res_num, [tf.shape(new_res_num)], message="new_res_num")
	  #max_num = tf.Print(max_num, [tf.shape(max_num)], message="max_num")
	   
	  finished = tf.where(tf.greater_equal(new_res_num, max_num),
					tf.ones_like(new_res_num, tf.bool),
					tf.zeros_like(new_res_num, tf.bool))

	  #finished = tf.reshape(tf.concat([tf.expand_dims(finished, 1)] * self._beam_size, 1), [-1])

	#tf.squeeze(cell_state.alignments, axis=1)
	
	#cell_outputs = tf.Print(cell_outputs, [tf.shape(cell_outputs)], message="rnnoutputs")
	#beam_parents = tf.Print(beam_parents, [tf.slice(beam_parents, [0], [96])], message="beamparent", summarize=100000)
	#beam_ends = tf.Print(beam_ends, [beam_ends], message="beamends", summarize=100000)
	#beam_end_parents = tf.Print(beam_end_parents, [beam_end_parents], message="beamendparents", summarize=1000000)
	#beam_end_probs = tf.Print(beam_end_probs, [tf.slice(beam_end_probs, [0], [192])], message="beamendendprobs", summarize=1000000)
	#beam_symbols = tf.Print(beam_symbols, [tf.slice(beam_symbols, [0], [96])], message="beamsymbols", summarize=1000000)

	beam_ends = tf.reshape(beam_ends, [-1, 2])
	beam_end_parents = tf.reshape(beam_end_parents, [-1, 2])
	beam_end_probs = tf.reshape(beam_end_probs, [-1, 2])
	outputs = BeamDecoderOutput(cell_outputs, beam_symbols, beam_parents, beam_ends, beam_end_parents, beam_end_probs, alignments)
	next_state = BeamState(beam_probs, next_state, new_res_num)

	#finished = tf.Print(finished, [tf.shape(finished)], message="fin", summarize=100000)
	#next_state = nest.map_structure(lambda x: tf.Print(x, [tf.shape(x)], message=str(x)+"_inbeam"), next_state) 

	return (outputs, next_state, next_inputs, finished)
