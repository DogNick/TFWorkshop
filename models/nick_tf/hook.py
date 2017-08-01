import os
import time

import numpy as np
import six

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache

def _as_graph_element(obj):
  """Retrieves Graph element."""
  graph = ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element

class NickSummaryHook(session_run_hook.SessionRunHook):
	def __init__(self,
				 summary_op,
				 debug_outputs_map,
				 output_dir=None,
				 sum_steps=20,
				 debug_steps=40):
		if sum_steps is not None and sum_steps <= 0:
			raise ValueError("invalid every_n_iter=%s." % sum_steps)
		if debug_steps is not None and debug_steps <= 0:
			raise ValueError("invalid every_n_iter=%s." % debug_steps)
		self._sum_steps = sum_steps
		self._timer_sum = SecondOrStepTimer(every_secs=None, every_steps=sum_steps) if sum_steps else None
		self._timer_dubug = SecondOrStepTimer(every_secs=None, every_steps=debug_steps) if debug_steps else None
		self._output_dir = output_dir
		self._summary_op = summary_op
		self._summary_writer = None
		self._debug_outputs_map = debug_outputs_map

	def begin(self):
		if self._summary_writer is None and self._output_dir:
			self._summary_writer = SummaryWriterCache.get(self._output_dir)
		self._next_step = None
		self._global_step_tensor = training_util.get_global_step()
		if self._global_step_tensor is None:
			raise RuntimeError(
				"Global step should be created to use SummarySaverHook.")
		self._iter_count = 0
		self._step_time = 0.0
		self._last_time = 0.0
		self._loss = 0.0

	def before_run(self, run_context):  # pylint: disable=unused-argument
		requests = {"global_steps":self._global_step_tensor}
		self._should_sum = self._timer_sum.should_trigger_for_step(self._iter_count)
		self._should_debug = self._timer_debug.should_trigger_for_step(self._iter_count)

		if self.timer_sum and self._should_sum and self._summary_op:
			requests["summary"] = self._summary_op
		if self.timer_debug and self._should_debug and self._debug_outputs_map:
			requests["debug_outputs"] = self._debug_outputs_map
		self._last_time = time.time()
		return SessionRunArgs(requests)

	def after_run(self, run_context, run_values):
		self._step_time += (time.time() - self._last_time) / self._sum_steps
		self._loss += run_values.results["loss"] / self._sum_steps
		self._iter_count += 1 
		if self._timer_sum and self._should_sum:
			ppx = math.exp(self._loss) if self._loss < 300 else float('inf')
			trainlg.info("[TRAIN] Global %d, Step-time %.2f, PPX %.2f" % (self._iter_count, self._step_time, ppx))
			self._step_time = self._loss = 0.0
			if "summary" in run_values.results:
		        for summary in run_values.results["summary"]:
					self._summary_writer.add_summary(summary, global_step)
		if self._timer_debug and self._should_debug:
			input_feed = run_context.original_args.feed_dict
			out = run_values.results["debug_outputs"]
			for idx in [5,10]:
				for key, value in input_feed.items():
					if isinstance(value[idx], list):
						value_str = " ".join([e for e in value[idx]])
					else:
						value_str = str(value[idx])
					trainlg.debug("[%s][%d] %s" % (key, idx, value_str))
					out_value_str = " ".join(list(out[idx]))
					trainlg.debug("[OUT][%d] %s" % (idx, out_value_str))
				trainlg.debug("")

	def end(self, session=None):
		if self._summary_writer:
			self._summary_writer.flush()


class NickCheckpointSaverHook(session_run_hook.SessionRunHook):
	def __init__(self,
				 checkpoint_dir,
				 checkpoint_steps,
				 saver,
				 feedfn=None,
				 dev_fetches=[], 
				 firein_steps=0,
				 checkpoint_basename="model.ckpt",
				 dev_n=5,
				 dev_batch_size=128,
				 listeners=None):
		
		logging.info("Create NickCheckpointSaverHook.")
		if saver is None:
			saver = saver_lib._get_saver_or_default()  # pylint: disable=protected-access
		self._saver = saver
		self._checkpoint_dir = checkpoint_dir
		self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
		self._timer = SecondOrStepTimer(every_secs=None, every_steps=save_steps)
		self._listeners = listeners or []
		self._firein_steps = firein_steps
		self._dev_fetches = dev_fetches
		self._feedfn = feedfn
		self._dev_n = dev_n
		self._dev_batch_size = dev_batch_size

	def begin(self):
		if self._global_step_tensor is None:
		    raise RuntimeError(
				"Global step should be created to use CheckpointSaverHook.")
		self._iter_count = 0
		for l in self._listeners:
			l.begin()
		self._prev_max_dev_loss = 10000


	def before_run(self, run_context):	
		if self._timer.last_triggered_step() is None:
			 # We do write graph and saver_def at the first call of before_run.
			 # We cannot do this in begin, since we let other hooks to change graph and
			 # add variables in begin. Graph is finalized after all begin calls.
			 training_util.write_graph(
				ops.get_default_graph().as_graph_def(add_shapes=True),
				self._checkpoint_dir,
				"graph.pbtxt")
			saver_def = self._get_saver().saver_def if self._get_saver() else None
			graph = ops.get_default_graph()
			meta_graph_def = meta_graph.create_meta_graph_def(
				    graph_def=graph.as_graph_def(add_shapes=True),
					saver_def=saver_def)
			self._summary_writer.add_graph(graph)
			self._summary_writer.add_meta_graph(meta_graph_def)
		requests = {"global_steps":self._global_step_tensor}	
		return SessionRunArgs(requests)

	def after_run(self, run_context, run_values):
		global_steps = run_values.results["global_steps"]
		self._should_trigger = self._timer.should_trigger_for_step(global_steps)
		if self._should_trigger:
			dev_loss = 0.0	
			dev_time = 0.0
			count = 0
			begin = 0
			sess = run_context.session
			while begin < self._dev_n: 
				input_feed = self._feed_fn(dev=True, begin, self._dev_batch_size, sess) 
				t0 = time.time()
				step_out = run_context.session.run(self._dev_fetches, input_feed)
				dev_loss += step_out["loss"] * 1.0 
				dev_time += round(time.time() - t0, 2)
				count += 1
				begin += self._dev_batch_size 
			dev_loss /= count 
			dev_time /= count
			summary = tf.Summary()
			summary.value.add(tag="dev_loss", simple_value=dev_loss)
			self.summary_writer.add_summary(summary, global_steps)
			dev_ppx = math.exp(dev_loss) if dev_loss < 300 else float('inf')
			trainlg.info("[Dev]Step-time %.2f, DEV_LOSS %.5f, DEV_PPX %.2f" % (dev_time, dev_loss, dev_ppx))
			self._timer.update_last_triggered_step(global_step)
			if global_steps > self._firein_steps and dev_loss < self._prev_max_dev_loss:
				trainlg.info("Need Saving....")
				self._prev_max_dev_loss = round(dev_loss, 4)
				self._saver.save(sess, self._save_path, global_step=global_steps)
