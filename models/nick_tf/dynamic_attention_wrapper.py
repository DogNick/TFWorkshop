# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A powerful dynamic attention wrapper object.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf

LSTMStateTuple = rnn_cell_impl.LSTMStateTuple


__all__ = [
    "DynamicAttentionWrapper",
    "DynamicAttentionWrapperState",
    "LuongAttention",
    "LuongAttentionNameScope",
    "BahdanauAttention",
    "hardmax",
]


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class AttentionMechanism(object):
  pass


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
  """Convert to tensor and possibly mask `memory`.

  Args:
    memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
    memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
    check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost
      dimensions are fully defined.

  Returns:
    A (possibly masked), checked, new `memory`.

  Raises:
    ValueError: If `check_inner_dims_defined` is `True` and not
      `memory.shape[2:].is_fully_defined()`.
  """
  memory = nest.map_structure(
      lambda m: ops.convert_to_tensor(m, name="memory"), memory)
  if check_inner_dims_defined:
    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError("Expected memory %s to have fully defined inner dims, "
                         "but saw shape: %s" % (m.name, m.get_shape()))
    nest.map_structure(_check_dims, memory)
  if memory_sequence_length is None:
    seq_len_mask = None
  else:
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=nest.flatten(memory)[0].dtype)
  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    if memory_sequence_length is not None:
      seq_len_mask = array_ops.reshape(
          seq_len_mask,
          array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
      return m * seq_len_mask
    else:
      return m
  return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.

  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self,
               query_layer,
               memory,
               max_mem_size,
               memory_sequence_length=None,
               memory_layer=None,
               check_inner_dims_defined=True,
               name=None):

    """Construct base AttentionMechanism class.

    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.



      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
        depth must match the depth of `query_layer`.
        If `memory_layer` is not provided, the shape of `memory` must match
        that of `query_layer`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
      name: Name to use when creating ops.
    """
    if (query_layer is not None
        and not isinstance(query_layer, layers_base._Layer)):  # pylint: disable=protected-access
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    if (memory_layer is not None
        and not isinstance(memory_layer, layers_base.Layer)):  # pylint: disable=protected-access
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    self._max_mem_size = max_mem_size
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)):
      self._values = _prepare_memory(
          memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys

class LuongAttentionNameScope(_BaseAttentionMechanism):
  """Implements Luong-style (multiplicative) attention scoring.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the normalized form.  This form is inspired by the
  normalization proposed for Bahdanau attention in

  Colin Raffel, Thang Luong, Peter J. Liu, Ron J. Weiss, and Douglas Eck.
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  (Eq. 15).

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self, num_units, memory, max_mem_size, memory_sequence_length=None,
               normalize=False, attention_r_initializer=None, z=None, scope=None,
               name="LuongAttentionNameScope"):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      attention_r_initializer:  Initial value of the post-normalization bias
        when normalizing.  Default is `0`.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    super(LuongAttentionNameScope, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(num_units, name="memory_layer"),
        memory=memory,
        max_mem_size=max_mem_size,
        memory_sequence_length=memory_sequence_length,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name
    self._scope = scope
    self._z = z
    if normalize and attention_r_initializer is None:
      attention_r_initializer = 0
    if normalize:
      with ops.name_scope(name, "LuongAttention",
                          [memory, attention_r_initializer]):
        attention_r_initializer = ops.convert_to_tensor(
            attention_r_initializer, dtype=self.values.dtype,
            name="attention_r_initializer")
    self._attention_r_initializer = attention_r_initializer

  def __call__(self, query, newmem=None):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.

    Returns:
      score: Tensor of dtype matching `self.values` and shape
        `[batch_size, max_time]` (`max_time` is memory's `max_time`).

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = int(query.get_shape()[-1])
    key_units = int(self.keys.get_shape()[-1])
    if depth != key_units:
      raise ValueError(
          "Incompatible or unknown inner dimensions between query and keys.  "
          "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
          "Perhaps you need to set num_units to the the keys' dimension (%s)?"
          % (query, depth, self.keys, key_units, key_units))
    dtype = query.dtype

    #with ops.name_scope(name, "LuongAttentionCall") as scope:
    with tf.variable_scope(self._name, "LuongAttentionNameScopeCall") as scope:
    #with tf.variable_scope(self._scope):
      # Reshape from [batch_size, depth] to [batch_size, 1, depth]
      # for matmul.

      # Inner product along the query units dimension.
      # matmul shapes: query is [batch_size, 1, depth] and
      #                keys is [batch_size, max_time, depth].
      # the inner product is asked to **transpose keys' inner shape** to get a
      # batched matmul on:
      #   [batch_size, 1, depth] . [batch_size, depth, max_time]
      # resulting in an output shape of:
      #   [batch_time, 1, max_time].
      # we then squeee out the center singleton dimension.
      #query = tf.Print(query, [query[0]], message="luo query", summarize=20)
      if newmem is not None:
          keys = tf.concat([self._keys, newmem], 1)
      else:
          keys = self._keys

      #keys = tf.Print(keys, [tf.shape(keys)], message="keys")
      if self._z != None:
          l = tf.shape(keys)[1]
          latent_size = int(self._z.get_shape()[1])

          # proj q and z to keys state size
          #q_w = variable_scope.get_variable("q_w", shape=[depth + int(self._z.get_shape()[-1]), depth], dtype=dtype)
          #query_z = tf.expand_dims(tf.matmul(tf.concat([query, self._z], 1), q_w), 1)
          #score = math_ops.matmul(query_z, keys, transpose_b=True)

          # proj keys to q_z size
          #with tf.variable_scope(scope):
          keys_kernel = variable_scope.get_variable("keys_kernel", shape=[1, 1, depth, depth + int(self._z.get_shape()[-1])], dtype=dtype)
          keys_for_score = tf.nn.conv2d(tf.expand_dims(keys, 1), keys_kernel, [1,1,1,1], "SAME", name="score")
          keys_for_score = tf.squeeze(keys_for_score)
          query_z = tf.expand_dims(tf.concat([query, self._z], 1), 1)
          score = math_ops.matmul(query_z, keys_for_score, transpose_b=True)
      else:
          query = array_ops.expand_dims(query, 1)
          score = math_ops.matmul(query, keys, transpose_b=True)
      score = array_ops.squeeze(score, [1])

      if self._normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / self._num_units)))
        # Scalar bias added to attention scores
        r = variable_scope.get_variable(
            "attention_r", dtype=dtype,
            initializer=self._attention_r_initializer)
        score = g * score + r
    #score = tf.Print(score, [score[0]], message="luo score", summarize=1000)

    return score

class LuongAttention(_BaseAttentionMechanism):
  """Implements Luong-style (multiplicative) attention scoring.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the normalized form.  This form is inspired by the
  normalization proposed for Bahdanau attention in

  Colin Raffel, Thang Luong, Peter J. Liu, Ron J. Weiss, and Douglas Eck.
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  (Eq. 15).

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self, num_units, memory, max_mem_size, memory_sequence_length=None,
               normalize=False, attention_r_initializer=None, z=None,
               name="LuongAttention"):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      attention_r_initializer:  Initial value of the post-normalization bias
        when normalizing.  Default is `0`.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    super(LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(num_units, name="memory_layer"),
        memory=memory,
        max_mem_size=max_mem_size,
        memory_sequence_length=memory_sequence_length,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name
    self._z = z
    if normalize and attention_r_initializer is None:
      attention_r_initializer = 0
    if normalize:
      with ops.name_scope(name, "LuongAttention",
                          [memory, attention_r_initializer]):
        attention_r_initializer = ops.convert_to_tensor(
            attention_r_initializer, dtype=self.values.dtype,
            name="attention_r_initializer")
    self._attention_r_initializer = attention_r_initializer

  def __call__(self, query, newmem=None):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.

    Returns:
      score: Tensor of dtype matching `self.values` and shape
        `[batch_size, max_time]` (`max_time` is memory's `max_time`).

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = int(query.get_shape()[-1])
    key_units = int(self.keys.get_shape()[-1])
    if depth != key_units:
      raise ValueError(
          "Incompatible or unknown inner dimensions between query and keys.  "
          "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
          "Perhaps you need to set num_units to the the keys' dimension (%s)?"
          % (query, depth, self.keys, key_units, key_units))
    dtype = query.dtype

    with ops.name_scope(None, "LuongAttentionCall", [query]):
      # Reshape from [batch_size, depth] to [batch_size, 1, depth]
      # for matmul.

      # Inner product along the query units dimension.
      # matmul shapes: query is [batch_size, 1, depth] and
      #                keys is [batch_size, max_time, depth].
      # the inner product is asked to **transpose keys' inner shape** to get a
      # batched matmul on:
      #   [batch_size, 1, depth] . [batch_size, depth, max_time]
      # resulting in an output shape of:
      #   [batch_time, 1, max_time].
      # we then squeee out the center singleton dimension.
      #query = tf.Print(query, [query[0]], message="luo query", summarize=20)

      if newmem is not None:
          keys = tf.concat([self._keys, newmem], 1)
      else:
          keys = self._keys

      #keys = tf.Print(keys, [tf.shape(keys)], message="keys")
      if self._z != None:
          l = tf.shape(keys)[1]
          latent_size = int(self._z.get_shape()[1])

          # proj q and z to keys state size
          #q_w = variable_scope.get_variable("q_w", shape=[depth + int(self._z.get_shape()[-1]), depth], dtype=dtype)
          #query_z = tf.expand_dims(tf.matmul(tf.concat([query, self._z], 1), q_w), 1)
          #score = math_ops.matmul(query_z, keys, transpose_b=True)

          # proj keys to q_z size
          keys_kernel = variable_scope.get_variable("keys_kernel", shape=[1, 1, depth, depth + int(self._z.get_shape()[-1])], dtype=dtype)
          keys_for_score = tf.nn.conv2d(tf.expand_dims(keys, 1), keys_kernel, [1,1,1,1], "SAME")
          keys_for_score = tf.squeeze(keys_for_score)
          query_z = tf.expand_dims(tf.concat([query, self._z], 1), 1)
          score = math_ops.matmul(query_z, keys_for_score, transpose_b=True)
      else:
          query = array_ops.expand_dims(query, 1)
          score = math_ops.matmul(query, keys, transpose_b=True)
      score = array_ops.squeeze(score, [1])

      if self._normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / self._num_units)))
        # Scalar bias added to attention scores
        r = variable_scope.get_variable(
            "attention_r", dtype=dtype,
            initializer=self._attention_r_initializer)
        score = g * score + r
    #score = tf.Print(score, [score[0]], message="luo score", summarize=1000)

    return score


class BahdanauAttention(_BaseAttentionMechanism):
  """Implements Bhadanau-style (additive) attention.

  This attention has two forms.  The first is Bhandanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form, Raffel attention, as described in:

  Colin Raffel, Thang Luong, Peter J. Liu, Ron J. Weiss, and Douglas Eck.
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  (Eq. 15).

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self, num_units, memory, max_mem_size, memory_sequence_length=None,
               normalize=False, attention_r_initializer=None, z=None,
               name="BahdanauAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      attention_r_initializer:  Initial value of the post-normalization bias
        when normalizing.  Default is `0`.
      name: Name to use when creating ops.
    """
    super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(num_units, name="query_layer"),
        memory_layer=layers_core.Dense(num_units, name="memory_layer"),
        memory=memory,
        max_mem_size=max_mem_size,
        memory_sequence_length=memory_sequence_length,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name
    if normalize and attention_r_initializer is None:
      attention_r_initializer = 0
    if normalize:
      with ops.name_scope(name, "BahdanauAttention",
                          [memory, attention_r_initializer]):
        attention_r_initializer = ops.convert_to_tensor(
            attention_r_initializer, dtype=self.values.dtype,
            name="attention_r_initializer")
    self._attention_r_initializer = attention_r_initializer

  def __call__(self, query, newmem=None):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
    Returns:
      score: Tensor of dtype matching `self.values` and shape
        `[batch_size, self.num_units]`.
    """
    with ops.name_scope(None, "BahndahauAttentionCall", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      dtype = processed_query.dtype
      # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
      processed_query = array_ops.expand_dims(processed_query, 1)
      v = variable_scope.get_variable(
          "attention_v", [self._num_units], dtype=dtype)
      if newmem is not None:
          keys = tf.concat([self._keys, newmem], 1)
      else:
          keys = self.keys
      if self._normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / self._num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [self._num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # Scalar bias added to attention scores
        r = variable_scope.get_variable(
            "attention_r", dtype=dtype,
            initializer=self._attention_r_initializer)
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        score = math_ops.reduce_sum(
            normed_v * math_ops.tanh(keys + processed_query + b), [2]) + r
      else:
        score = math_ops.reduce_sum(
            v * math_ops.tanh(keys + processed_query), [2])

    return score


class DynamicAttentionWrapperState(
    collections.namedtuple(
        "DynamicAttentionWrapperState", ("cell_state", "attention", "newmem", "alignments"))):
  """`namedtuple` storing the state of a `DynamicAttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell`.
    - `attention`: The attention emitted at the previous time step.
  """
  pass


def hardmax(logits, name=None):
  """Returns batched one-hot vectors.

  The depth index containing the `1` is that of the maximum logit value.

  Args:
    logits: A batch tensor of logit values.
    name: Name to use when creating ops.
  Returns:
    A batched one-hot tensor.
  """
  with ops.name_scope(name, "Hardmax", [logits]):
    logits = ops.convert_to_tensor(logits, name="logits")
    if logits.get_shape()[-1].value is not None:
      depth = logits.get_shape()[-1].value
    else:
      depth = array_ops.shape(logits)[-1]
    return array_ops.one_hot(
        math_ops.argmax(logits, -1), depth, dtype=logits.dtype)


class DynamicAttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_size,
               cell_input_fn=None,
               probability_fn=None,
               output_attention=True,
               name=None,
               addmem=False,
               see_attn_scores=False):
    """Construct the `DynamicAttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_size: Python integer, the depth of the attention (output)
        tensor.
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      name: Name to use when creating ops.
    """
    if not isinstance(cell, rnn_cell_impl.RNNCell):
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "probability_fn must be callable, saw type: %s"
            % type(probability_fn).__name__)
    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._attention_size = attention_size
    self._attention_layer = layers_core.Dense(
        attention_size, bias_initializer=None)
    self._cell_input_fn = cell_input_fn
    self._probability_fn = probability_fn
    self._output_attention = output_attention
    self._addmem = addmem
    self._name = name


  @property
  def state_shape(self):
    return DynamicAttentionWrapperState(
            cell_state=nest.map_structure(lambda s: tf.TensorShape([None, s]), self._cell.state_size),
                attention=tf.TensorShape([None, self._attention_size]),
                alignments=tf.TensorShape([None, 1, None]),
                newmem=tf.TensorShape([]))
  @property
  def output_size(self):
    return self._attention_size


  @property
  def state_size(self):
    return DynamicAttentionWrapperState(
                cell_state=self._cell.state_size,

                attention=self._attention_size,
                alignments=self._attention_mechanism._max_mem_size,
                newmem=self._attention_size)

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      zero_newmem = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
      zero_newmem = zero_newmem.write(0, tf.zeros([batch_size, self._attention_size]))
      return DynamicAttentionWrapperState(
          cell_state=self._cell.zero_state(batch_size, dtype),
          attention=_zero_state_tensors(
              self._attention_size, batch_size, dtype),
          alignments=tf.zeros([batch_size, 1, self._attention_mechanism._max_mem_size]),
          newmem=zero_newmem)


  def __call__(self, inputs, state, scope=None):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer.


    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `DynamicAttentionWrapperState` containing
        tensors from the previous time step.
      scope: Must be `None`.

    Returns:
      A tuple `(attention, next_state)`, where:

      - `attention` is the attention passed to the layer above.
      - `next_state` is an instance of `DynamicAttentionWrapperState`
         containing the state calculated at this time step.
    Raises:
    """
    if scope is not None:
      raise NotImplementedError("scope not None is not supported")

    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state


    cell_output, next_cell_state = self._cell(cell_inputs, cell_state, self._name)

    #cell_output = tf.Print(cell_output, [tf.shape(cell_output)], message="cell_output") 
    # Nick use cell_state to query attention
    if isinstance(next_cell_state[0], LSTMStateTuple):
        state_to_query = next_cell_state[-1].h 
    else:
        state_to_query = next_cell_state[-1]

    if self._addmem:
        score = self._attention_mechanism(state_to_query, tf.transpose(state.newmem.stack(), [1, 0, 2]))
    else:
        score = self._attention_mechanism(state_to_query)

    #score = self._attention_mechanism(cell_output)
    alignments = self._probability_fn(score)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, attention_mechanism.num_units]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, attention_mechanism.num_units].
    # we then squeeze out the singleton dim.


    if self._addmem:
        context = math_ops.matmul(alignments, tf.concat([self._attention_mechanism.values, tf.transpose(state.newmem.stack(), [1, 0, 2])], 1))
    else:
        context = math_ops.matmul(alignments, self._attention_mechanism.values)

    context = array_ops.squeeze(context, [1])
    attention = self._attention_layer(
        array_ops.concat([cell_output, context], 1))

    if self._output_attention:
        cell_output = attention

    #if self._addmem:
    #    #newmem = tf.concat([state.newmem, tf.expand_dims(cell_output, 1)], 1)
    #    newmem = state.newmem.write(state.newmem.size(), cell_output)
    #else:
    #    newmem = state.newmem
    newmem = state.newmem

    batch_size = tf.shape(alignments)[0]
    pad_size = self._attention_mechanism._max_mem_size - tf.shape(alignments)[2]
    alignments = tf.concat([alignments, tf.zeros([batch_size, 1, pad_size])], 2)
     

    next_state = DynamicAttentionWrapperState(

           cell_state=next_cell_state,
           attention=attention,
           alignments=alignments,
           newmem=newmem)

    return cell_output, next_state
