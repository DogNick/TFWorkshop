#coding=utf-8
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
define("schedule", default=0, help="server schedule id", type=int)

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

from DeepMatchHandler import *
from GenerateHandler import *
from PoemHandler import *


def main():
	# start tf_model_servers
	DEPLOY_ROOT = options.deploy_root 
	NUM_BATCH_THREADS = options.num_batch_threads 
	BATCH_TIMEOUT_MICROS = options.batch_timeout_micros 
	MAX_ENQUEUED_BATCHES = options.max_enqueued_batches 
	MAX_BATCH_SIZE = options.max_batch_size 

	model_infos = {} 
	for each in schedule:
		host, port = schedule[each]["tf_server"].split(":")
		gpu = schedule[each]["deploy_gpu"]
		model_infos[each] = (each, host, port, gpu)

	if options.submodels != "":
		submodel_names = options.submodels.split(",")
		for name in submodel_names:
			if name not in model_infos:
				serverlg.error("Error %s not in SERVER_SCHEDULES !" % name)
				exit(0)
			submodel_infos[name] = model_infos[name]
	else:
		submodel_infos = model_infos

	children = []
	for key in submodel_infos:
		env = os.environ.copy()
		runtime_name, host, port, gpu = model_infos[key] 
		env["CUDA_VISIBLE_DEVICES"] = str(gpu)
		command = "./tensorflow_model_server --enable_batching=true" \
				  " --port=%d" \
				  " --model_name=%s" \
				  " --model_base_path=%s/%s" \
				  " --num_batch_threads=%d" \
				  " --batch_timeout_micros=%d" \
				  " --max_enqueued_batches=%d" \
				  " --max_batch_size=%d &" % (
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

	#for child in children:
	#	child.wait()

	def signal_handler(signum, frame):
		for child in children:
			os.killpg(os.getpgid(child.pid), signum)
			print('You killed  child %d' % child.pid)
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)
	HandlerMap = {
			"tsinghua": TsinghuaHandler,
			"generate": GenerateHandler,
			"matchpoem": MatchPoemHandler,
			"judgepoem": JudgePoemHandler,
			"intent": ModelHandler
	}

	# start front servers 
	try:
		app = tornado.web.Application(
				[
					(r'/%s' % options.service, HandlerMap[options.service]),
				]
		)
		server = tornado.httpserver.HTTPServer(app)
		server.bind(options.port)
		server.start(options.procnum)
		serverlg.info("[SERVICE START] Generate adptor server start, listen on %d" % options.port)
		tornado.ioloop.IOLoop.instance().start()
	except:
		for child in children:
			os.killpg(os.getpgid(child.pid), signal.SIGTERM)
			serverlg.warning('You killed  child %d' % child.pid)

if __name__ == "__main__":
	main()
