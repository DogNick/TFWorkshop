#coding=utf-8
import itertools
from ModelHandler import *
from models import util
from models.Nick_plan import score_with_prob_attn

class ChatENHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
		schedule = self.schedule
		query = self.get_argument('query', None)
		n = self.get_argument('n', 10)
		self.debug = self.get_argument("debug", "0")
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()

		results = []
		debug_infos = []

		generators = {} 
		scorers = {} 
		for idx, model_conf in enumerate(schedule["servables"]):
			name = model_conf["model"]
			if name.find("_reverse") == -1:
				generators[idx] = conf["graph_stub"]
			else:
				scorers[idx] = conf["graph_stub"]
				lmda = float(self.get_argument("r", model_conf.get("lmda", 0.5)))
				a = float(self.get_argument("a", model_conf.get("alpha", 0.7)))
				b = float(self.get_argument("b", model_conf.get("beta", 0.1)))
		
		multi_models = []
		for idx, graph_stub in generators.items():
			g, stub = graph_stub
			joint_prime = schedule["servables"][idx].get("joint_prime", False)
			query_cleaned = util.clean_en(query.encode("utf-8"), joint_prime=joint_prime)
			serverlg.info('[chatbot] [QUERY CLEAN] [%s] [%s] [%s]' % (schedule["servables"][idx]["model"], time.strftime('%Y-%m-%d %H:%M:%S'), query_cleaned))
			multi_models.append(self.run_model(g, stub, [query_cleaned], use_seg=False))

		outs = yield multi(multi_models)

		# collect results of one example from all models, for one request, only one example here
		# batch_size = 1

		#exmaple_results = itertools.chain.from_iterable(zip(*outs)[0])
		example_results = []
		[ example_results.extend(res) for res in zip(*outs)[0]]

		outputs = [each["outputs"] for each in example_results]
		probs = [each["probs"] for each in example_results]
		names = [each["model_name"] for each in example_results]

		# for log posterior probability 
		pairs = ["%s\t%s" % (query_cleaned, " ".join(each)) for each in outputs]
		multi_models = [self.run_model(g, stub, pairs, use_seg=False) for g, stub in scorers.values()]
		multi_model_posteriors = yield multi(multi_models)

		posteriors = []
		for each in zip(*multi_model_posteriors):
			sum_posterior = 0.0	
			for res in each:
				sum_posterior += res["posteriors"]
			posteriors.append(sum_posterior)
				
		# score results based on likelihood, posterior, valid check, and gnmt criteria 
		scored = score_with_prob_attn(outputs, probs, None, posteriors, lbda=lmda, alpha=a, beta=b, is_ch=False, average_across_len=False)  

		for i, name in enumerate(names):
			final, infos = scored[i]
			infos["model_name"] = name
			infos["r"] = lmda
			infos["a"] = a 
			infos["b"] = b 
			scored[i] = (final, infos)

		sorted_res = sorted(scored, key=lambda x:x[1]["score"], reverse=True)

		n = len(sorted_res) if n > len(sorted_res) else n
		results = [each[0] for each in sorted_res[0:n]] 
		debug_infos = [each[1] for each in sorted_res[0:n]]

		raise gen.Return((results, debug_infos, schedule["desc"]))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			if self.debug == "0":
				multi_results.append({"answer":each})
			else:
				multi_results.append({"answer":each, "debug_info":infos[i]})
		random.shuffle(multi_results)
		return multi_results

class GenerateHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
		schedule = self.schedule
		query = self.get_argument('query', None)
		n = self.get_argument('n', 10)
		self.debug = self.get_argument("debug", "0")
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()

		results = []
		debug_infos = []

		generators = {} 
		scorers = {} 
		for idx, model_conf in enumerate(schedule["servables"]):
			name = model_conf["model"]
			if name.find("_reverse") == -1:
				generators[idx] = model_conf["graph_stub"]
			else:
				scorers[name] = model_conf["graph_stub"]

		multi_models = []
		query_utf8 = query.encode("utf-8")
		for idx, graph_stub in generators.items():
			g, stub = graph_stub
			serverlg.info('[chatbot] [QUERY CLEAN] [%s] [%s] [%s]' % (schedule["servables"][idx]["model"], time.strftime('%Y-%m-%d %H:%M:%S'), query_utf8))
			multi_models.append(self.run_model(g, stub, [query_utf8], use_seg=True))
		# collect results of one example from all models, for one request, only one example here
		# batch_size = 1
		outs = yield multi(multi_models)


		#exmaple_results = itertools.chain.from_iterable(zip(*outs)[0])
		example_results = []
		[ example_results.extend(res) for res in zip(*outs)[0]]

		outputs = [each["outputs"] for each in example_results]
		probs = [each["probs"] for each in example_results]
		names = [each["model_name"] for each in example_results]

		# for log posterior probability 
		pairs = ["%s\t%s" % (query_utf8, " ".join(each)) for each in outputs]
		multi_models = []
		for idx, graph_stub in scorers.items():
			g, stub = graph_stub
			multi_models.append(self.run_model(g, stub, pairs, use_seg=False))
		multi_model_posteriors = yield multi(multi_models)
		
		posteriors = []
		for each in zip(*multi_model_posteriors):
			sum_posterior = 0.0	
			for res in each:
				sum_posterior += res["posteriors"]
			posteriors.append(sum_posterior)
		
		# score results based on likelihood, posterior, valid check, and gnmt criteria 
		params = schedule.get("params", {})
		lmda = float(self.get_argument("r", params.get("lmda", 0.5)))
		a = float(self.get_argument("a", params.get("alpha", 0.6)))
		b = float(self.get_argument("b", params.get("beta", 0.1)))

		scored = score_with_prob_attn(outputs, probs, None, posteriors, lbda=lmda, alpha=a, beta=b, is_ch=False, average_across_len=False)  

		for i, name in enumerate(names):
			final, infos = scored[i]
			infos["model_name"] = name
			infos["r"] = lmda
			infos["a"] = a 
			infos["b"] = b 
			scored[i] = (final, infos)

		sorted_res = sorted(scored, key=lambda x:x[1]["score"], reverse=True)

		n = len(sorted_res) if n > len(sorted_res) else n
		results = [each[0] for each in sorted_res[0:n]] 
		debug_infos = [each[1] for each in sorted_res[0:n]]

		raise gen.Return((results, debug_infos, schedule.get("desc", "no description")))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			if self.debug == "0":
				multi_results.append({"answer":each})
			else:
				multi_results.append({"answer":each, "debug_info":infos[i]})
		random.shuffle(multi_results)
		return multi_results
