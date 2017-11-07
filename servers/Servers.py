#coding=utf-8
import tornado
import tornado.httpserver
import tornado.options
import tornado.ioloop

from tornado.options import define, options, parse_command_line
from tornado.ioloop import IOLoop

import subprocess
import logging
import signal
import time
import sys
import os
import gc

define('port',default=9000,help='run on the port',type=int)
define('procnum',default=32,help='process num',type=int)

define("deploy_root", default="deployments", help="", type=str)
define("submodels", default="", help="", type=str)
define("num_batch_threads", default=32, help="", type=int)
define("batch_timeout_micros", default=30000, help="", type=int)
define("max_enqueued_batches", default=32, help="", type=int)
define("max_batch_size", default=64, help="", type=int)

define("service", default="generate", help="server name", type=bytes)
define("schedule", default=None, help="server schedule id", type=bytes)

parse_command_line()

serverlg = logging.getLogger("")

#serverlg = logging.getLogger("server")
#fh = logging.FileHandler(os.path.join("logs", "Server_%s_%d_%d.log" % (options.service, options.schedule, options.port)))
#fh.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
#serverlg.setLevel(logging.DEBUG)
#serverlg.addHandler(fh)
#
#exclg = logging.getLogger("exception")
#fh = logging.FileHandler(os.path.join("logs", "exc_%d.log" % options.port))
#fh.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
#exclg.setLevel(logging.DEBUG)
#exclg.addHandler(fh)

from GenerateHandler import *
from PoemHandler import *
from IntentHandler import * 
from ScoreHandler import * 


def main():
	# start tf_model_servers
	DEPLOY_ROOT = options.deploy_root 
	NUM_BATCH_THREADS = options.num_batch_threads 
	BATCH_TIMEOUT_MICROS = options.batch_timeout_micros 
	MAX_ENQUEUED_BATCHES = options.max_enqueued_batches 
	MAX_BATCH_SIZE = options.max_batch_size 

	# Multi service support is coming soon... 
	if options.service not in SERVICE_SCHEDULES:
		raise ValueError("service name %s not in SERVICE_SCHEDULES" % options.service)
		exit(1)
	if options.schedule not in SERVICE_SCHEDULES[options.service]:
		raise ValueError("schedule name %s not in %s service defined in SERVICE_SCHEDULES" % (options.schedule, options.service))
		exit(1)

	schedule = SERVICE_SCHEDULES[options.service][options.schedule]
	serverlg.info('[Graph and servable stubs] [Initialization: service %s, schedule %s] [%s]' % 
					(options.service, options.schedule, time.strftime('%Y-%m-%d %H:%M:%S')))

	# Build model infos of (graph, service_stub) tuple with its model conf and service conf
	model_infos = {} 
	for i, model_conf in enumerate(schedule["servables"]):
		host, port = model_conf["tf_server"].split(":")
		channel = implementations.insecure_channel(host, int(port))
		stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

		name = model_conf["model"]
		gpu = model_conf["deploy_gpu"]
		graph = models.create(name)
		graph.apply_deploy_conf(model_conf)
		schedule["servables"][i]["graph_stub"] = (graph, stub) 
		model_infos[name] = (name, host, port, gpu)

	# Start tensorflow-serving backends according to parsed model_infos
	children = []
	for key in model_infos:
		env = os.environ.copy()
		runtime_name, host, port, gpu = model_infos[key] 
		env["CUDA_VISIBLE_DEVICES"] = str(gpu)
		env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
		command = "./tensorflow_model_server --enable_batching=true" \
				  " --port=%d" \
				  " --model_name=%s" \
				  " --model_base_path=%s/%s" \
				  " --num_batch_threads=%d" \
				  " --batch_timeout_micros=%d" \
				  " --max_enqueued_batches=%d" \
				  " --max_batch_size=%d" % (
					  int(port), 
					  runtime_name, 
					  DEPLOY_ROOT, runtime_name, 
					  NUM_BATCH_THREADS, 
					  BATCH_TIMEOUT_MICROS, 
					  MAX_ENQUEUED_BATCHES, 
					  MAX_BATCH_SIZE)
		log = open("logs/tf_server_%s_%d.log" % (runtime_name, int(port)), "w+") 
		child = subprocess.Popen(command.split(), stdout=log, stderr=log, preexec_fn=os.setpgrp, env=env)
		serverlg.info("Start tensorflow model server [%s] %s:%d gpu:%s" % (runtime_name, host, int(port), str(gpu)))
		children.append(child)

	def signal_handler(signum, frame):
		for child in children:
			os.killpg(os.getpgid(child.pid), signum)
			print('You killed  child %d' % child.pid)
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)
	HandlerMap = {
			"generate": GenerateHandler,
			"matchpoem": MatchPoemHandler,
			"judgepoem": JudgePoemHandler,
			"intent": IntentHandler,
			"chaten": ChatENHandler, 
			"scorer": ScoreHandler 
	}

	# start front servers 
	try:
		app = tornado.web.Application(
				[
					(r'/%s' % options.service, HandlerMap[options.service], dict(schedule=schedule)),
				]
		)

		server = tornado.httpserver.HTTPServer(app)
		server.bind(options.port)
		server.start(options.procnum)
		serverlg.info("[SERVICE START] Generate adptor server start, listen on %d" % options.port)
		tornado.ioloop.IOLoop.instance().start()
	except Exception, e:
		serverlg.error("[EXCEPTION] %s" % traceback.format_exc(e))
		for child in children:
			os.killpg(os.getpgid(child.pid), signal.SIGTERM)
			serverlg.warning('You killed  child %d' % child.pid)

if __name__ == "__main__":
	main()
