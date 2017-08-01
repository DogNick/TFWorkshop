# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import tensorflow as tf
#from tensorflow.contrib.seq2seq.python.ops import decoder
#from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
import decoder
import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops


__all__ = [
    "ScoreDecoderOutput",
    "ScoreDecoder",
]


class ScoreDecoderOutput(
    collections.namedtuple("ScoreDecoderOutput", ("logprobs"))):
  pass


class ScoreDecoder(decoder.Decoder):
  """Score decoder."""

  def __init__(self, cell, helper, initial_state, out_proj, output_layer=None):

    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._out_proj = out_proj
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  @property
  def output_shape(self):
      return ScoreDecoderOutput(
        logprobs=tensor_shape.TensorShape([]))

  @property
  def zero_outputs(self):
    def _create_zero_outputs(size, dtype, batch_size):
        """Create a zero outputs Tensor structure."""
        def _t(s):
          return (s if isinstance(s, ops.Tensor) else constant_op.constant(
              tensor_shape.TensorShape(s).as_list(),
              dtype=dtypes.int32,
              name="zero_suffix_shape"))
      
        def _create(s, d):
          return array_ops.zeros(
              array_ops.concat(
                  ([batch_size], _t(s)), axis=0), dtype=d)

        return nest.map_structure(_create, size, dtype)

    return _create_zero_outputs(self.output_size,
                                self.output_dtype,
                                self.batch_size)
  @property
  def state_shape(self):
      return self._cell.state_shape


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
    return ScoreDecoderOutput(
        logprobs=tensor_shape.TensorShape([self._out_proj[0].shape[1]]))
    
  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_state)[0].dtype
    return ScoreDecoderOutput(dtypes.float32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
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
    with ops.name_scope(name, "ScoreDecoderStep", (time, inputs, state)):
      w, b = self._out_proj 
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
        
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)

      #row_idx = tf.cumsum(tf.ones([self.batch_size, 1], tf.int32), axis=0, exclusive=True)
      #col_idx = tf.expand_dims(sample_ids, 1)
      #idx = tf.concat([row_idx, col_idx], 1)
      logprobs = tf.nn.log_softmax(tf.nn.xw_plus_b(cell_outputs, w, b), 1)

      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)


      
    outputs = ScoreDecoderOutput(logprobs)
    return (outputs, next_state, next_inputs, finished)
