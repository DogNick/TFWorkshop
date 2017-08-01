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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.util import nest
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

_linear = core_rnn_cell_impl._linear

#__all__ = [
#    "AttnTopicDecoderOutput",
#    "AttnTopicDecoder"
#]

class AttnTopicDecoderOutput(
    collections.namedtuple("AttnTopicDecoderOutput", ("rnn_output", "sample_id"))):
  pass

class AttnTopicDecoder(decoder.Decoder):
    """AttnTopicDecoder decoder."""
    def __init__(self, cell, helper, initial_state, attn_states, attn_size,
                  topic_attn_states=None, topic_attn_size=None, attn_vec_size=None, output_layer=None):
      """Initialize AttnTopicDecoder.

      Args:
        cell: An `RNNCell` instance.
        helper: A `Helper` instance.
        initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        attn_states: A tensor of shape [batch_size, attention_length, attention_size] to do attention over.
        attn_vec_len: (Optional) intermediate size for attention action, is the same with that of input attn_states as defualt 
        topic_attn_states: (Optional) some other tensor of the same shape with attn_states to do attention over if not None.
        output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
          `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
          to storing the result or sampling.

      Raises:
        TypeError: if `cell` is not an instance of `RNNCell`, `helper`
          is not an instance of `Helper`, or `output_layer` is not an instance
          of `tf.layers.Layer`.
      """
      if not isinstance(cell, core_rnn_cell.RNNCell):
        raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
      if not isinstance(helper, helper_py.Helper):
        raise TypeError("helper must be a Helper, received: %s" % type(helper))
      if (output_layer is not None
          and not isinstance(output_layer, layers_base._Layer)):  # pylint: disable=protected-access
        raise TypeError(
            "output_layer must be a Layer, received: %s" % type(output_layer))

      self._cell = cell
      self._helper = helper

      self._initial_state = initial_state
      self._output_layer = output_layer

      self._attn_size = attn_size
      self._attn_states = attn_states 
      self._topic_attn_states = topic_attn_states 
      self._topic_attn_size = topic_attn_size 
      self._attn_vec_size = attn_size if attn_vec_size is None else attn_vec_size

    @property
    def batch_size(self):
        return self._helper.batch_size

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
        return AttnTopicDecoderOutput(rnn_output=self._rnn_output_size(), sample_id=tensor_shape.TensorShape([]))

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and int32 (the id)
        dtype = nest.flatten(self._initial_state)[0].dtype
        return AttnTopicDecoderOutput(nest.map_structure(lambda _: dtype, self._rnn_output_size()), dtypes.int32)

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
          name: Name scope for any created operations.

        Returns:
          `(finished, first_inputs, initial_state)`.
        """
        if self._topic_attn_states != None: 
            attns, attn_states = self._attention(self._initial_state, self._attn_states,
                                                self._attn_size, self._attn_vec_size, "attention")
            topic_attns, topic_attn_states = self._attention(self._initial_state, self._topic_attn_states,
                                                self._topic_attn_size, self._attn_vec_size, "topic_attention")
            self._initial_state = (self._initial_state, attns, attn_states, topic_attns, topic_attn_states)
        else:
            attns, attn_states, self._attention(self._initial_state, self._attn_states,
                                                self.attn_size, self_attn_vec_size, "attention")
            self._initial_state = (self._initial_state, attns, attn_states)
        return self._helper.initialize() + (self._initial_state,)

    #def _attention2(self, query, attn_states):

    def _attention(self, query, attn_states, attn_size, attn_vec_size, scope, reuse=None):
        conv2d = nn_ops.conv2d
        reduce_sum = math_ops.reduce_sum
        softmax = nn_ops.softmax
        tanh = math_ops.tanh
        attn_len = tf.shape(attn_states)[1] 
        with vs.variable_scope(scope or "attention", reuse=reuse):
          k = vs.get_variable("attn_w", [1, 1, attn_size, attn_vec_size])
          v = vs.get_variable("attn_v", [attn_vec_size])
          hidden = tf.expand_dims(attn_states, 2)
          #hidden = array_ops.reshape(attn_states, [-1, self._attn_length, 1, attn_size])

          hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
          y = _linear(query, attn_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attn_vec_size])
          s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
          a = softmax(s)
          d = reduce_sum(array_ops.reshape(a, [-1, attn_len, 1, 1]) * hidden, [1, 2])

          new_attns = array_ops.reshape(d, [-1, attn_size])
          # Nick thinks sliding window is not necessary here
          #new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
          return new_attns, attn_states

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
        if self._topic_attn_states != None:
            state, attns, attn_states, topic_attns, topic_attn_states = state
        else:
            state, attns, attn_states = state

        input_size = inputs.get_shape().as_list()[1]

        # Use or not
        with ops.name_scope(name, "AttnTopicDecoderStep", (time, inputs, state)):
            # merge current inputs with previous attns
            if self._topic_attn_states != None:
                with vs.variable_scope("input"):
                    inputs = _linear([inputs, attns, topic_attns], input_size, True)
            else:
                with vs.variable_scope("input"):
                    inputs = _linear([inputs, attns], input_size, True)

            # call rnn cell to update state
            cell_outputs, cell_state = self._cell(inputs, state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            new_state_cat = array_ops.concat(nest.flatten(cell_state), 1)

            # Create new attns based on new state
            if self._topic_attn_states != None:
                new_attns, new_attn_states = self._attention(new_state_cat, attn_states, self._attn_size,
                                                            self._attn_vec_size, "attention", True)
                new_topic_attns, new_topic_attn_states = self._attention(new_state_cat, topic_attn_states,
                                                                self._topic_attn_size, self._attn_vec_size, "topic_attention", True)

                new_state = (cell_state, new_attns, new_attn_states, new_topic_attns, new_topic_attn_states)
            else:
                new_attns, new_attn_states = self._attention(new_state_cat, attn_states, self._attn_size,
                                                                self._attn_vec_size, "attention", True)
                new_state = (cell_state, new_attns, new_attn_states)

            if self._topic_attn_states != None:
                with vs.variable_scope("output"):
                    outputs = _linear([cell_outputs, new_attns, new_topic_attns], self._attn_size, True)
            else:
                with vs.variable_scope("output"):
                    outputs = _linear([cell_outputs, new_attns], self._attn_size, True)

            #new_state = array_ops.concat(list(new_state), 1)

            sample_ids = self._helper.sample(time=time, outputs=outputs, state=new_state)

            (finished, next_inputs, next_state) = self._helper.next_inputs(time=time, outputs=outputs,
                                                            state=new_state, sample_ids=sample_ids)

        outputs = AttnTopicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)
if __name__ == "__main__":
    pass
