#a.sys
import abc
import gc
import json
import logging
import random
import re
import sys
import time
import urllib

#b.outer
from grpc.beta import implementations
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util
import tornado.web
import tornado.gen
from tornado.options import define, options, parse_command_line
from tornado.locks import Event
from tornado import gen, locks
from tornado.gen import multi
from tornado.gen import Future
from tornado.ioloop import IOLoop

#c.self
sys.path.insert(0, "..")
##############################################################################################################
# use just everything same with that used during training in models/ to keep the INHERENT preproc and afterproc
#############################################################################################################
import models
from service_schedules import SERVICE_SCHEDULES

serverlg = logging.getLogger("")

def _fwrap(f, gf):
	try:
		f.set_result(gf.result())
	except Exception as e:
		f.set_exception(e)

def fwrap(gf, ioloop=None):
	'''
		Wraps a GRPC result in a future that can be yielded by tornado
		Usage::
																
		@coroutine
		def my_fn(param):
			result = yield fwrap(stub.function_name.future(param, timeout))
	'''
	f = Future()
	if ioloop is None:
		ioloop = IOLoop.current()

	gf.add_done_callback(lambda _: ioloop.add_callback(_fwrap, f, gf))
	return f


class ModelHandler(tornado.web.RequestHandler):
	__metaclass__ = abc.ABCMeta
	servables = [] 
	def initialize(self, schedule):
		self.schedule = schedule 
	
	@abc.abstractmethod
	def handle(self): 
		query = self.get_argument('query', None)
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()
		results = []
		debug_infos = []
		 
		graph_stubs = [model_conf["graph_stub"] for model_conf in schedule["servables"]]
		# Multi model compatible, but here just one model exists
		multi_models = []
		for graph, stub in graph_stubs:
			multi_models.append(self.run_model(graph, stub, [query.encode("utf-8")]))
		outs = yield multi(multi_models)
		serverlg.info(outs)
			
		raise gen.Return(None)	

	@abc.abstractmethod
	def form_multi_results(self, model_name, model_out): 
		return

	def set_default_header(self):
		self.set_header('Access-Control-Allow-Origin', "*")

	@tornado.gen.coroutine
	def run_model(self, graph, stub, records, use_seg=True):
		# Use model specific preprocess
		feed_data = graph.preproc(records, use_seg=use_seg, for_deploy=True)
		# make request 
		request = predict_pb2.PredictRequest()
		request.model_spec.name = graph.name 
		values = feed_data.values()	
		N = 100 
		keys = feed_data.keys()
		# Just for logging 
		for i, example in enumerate(zip(*feed_data.values())):
			if i == N:
				break
			for j, v in enumerate(example):
				serverlg.info('[DispatcherServer] [%d][%s] %s' % (i, keys[j], str(v)))

		# Form model request
		for key, value in feed_data.items():
			v = np.array(value) 
			value_tensor = tensor_util.make_tensor_proto(value, shape=v.shape)
			# For compatibility to the old placeholder key 
			request.inputs[key].CopyFrom(value_tensor)

		# query the model 
		#result = stub.Predict(request, 4.0)
		result = yield fwrap(stub.Predict.future(request, 3.0))
		out = {}
		for key, value in result.outputs.items():
			out[key] = tensor_util.MakeNdarray(value) 
		model_results = graph.after_proc(out)

		raise gen.Return(model_results)
		
	@tornado.web.asynchronous
	@tornado.gen.coroutine
	def post(self):
		serverlg.info('[DispatcherServer] [BEGIN] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
		gc.disable()

		# prepare locks, events, and results container for coroutine 
		#self.results = [None] * len(deployments)
		#self.model_results = []
		#self.evt = Event()
		#self.lock = locks.Lock()
		
		# query all models 
		#for name in self.servings:
		model_results, debug_infos, desc = yield self.handle()
		results = self.form_multi_results(model_results, debug_infos)

		# wait until told to proceed
		#yield self.evt.wait()

		#self.run()

		# form response
		ret = {"status":"ok","result":results}
		#self.write(json.dumps(ret))
		self.write(ret)
		#self.finish()

	@tornado.web.asynchronous
	@tornado.gen.coroutine
	def get(self):
		gc.disable()
		serverlg.info('[DispatcherServer] [BEGIN] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
		# preproc 
		model_results, debug_infos, desc = yield self.handle()
		results = self.form_multi_results(model_results, debug_infos)

		# form response
		ret = {"status":"ok","result":results, "desc":desc}
		self.write(json.dumps(ret, ensure_ascii=False))
