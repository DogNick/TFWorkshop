#coding=utf-8
import itertools
from ModelHandler import *
class ChatENHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
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
		for name in schedule:
			if name.find("_reverse") == -1:
				generators[name] = schedule[name]["graph_stub"]
			else:
				scorers[name] = schedule[name]["graph_stub"]
				lmda = float(self.get_argument("r", schedule[name].get("lmda", 0.5)))
				a = float(self.get_argument("a", schedule[name].get("alpha", 0.7)))
				b = float(self.get_argument("b", schedule[name].get("beta", 0.1)))

		joint_prime = schedule[name].get("joint_prime", False) 
		serverlg.info(query.encode("utf-8"))
		query_cleaned = util.clean_en(query.encode("utf-8"), joint_prime=joint_prime)
		serverlg.info('[chatbot] [QUERY CLEAN] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), query_cleaned))

		multi_models = [self.run_model(g, stub, [query_cleaned], use_seg=False) for g, stub in generators.values()]
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
		posteriors = [sum(each) for each in zip(*multi_model_posteriors)]
		
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

		raise gen.Return((results, debug_infos, DESC["chaten"]))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			if self.debug == "0":
				multi_results.append({"answer":each})
			else:
				multi_results.append({"answer":each, "debug_info":infos[i]})
		random.shuffle(multi_results)
		return multi_results


class TsinghuaHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
		query = self.get_argument('query', None)
		n = self.get_argument('n', 5)
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()

		results = []
		debug_infos = []

		words = util.tokenize_word(query.encode("utf-8"))
		# beam_search generate 
		graph, stub = schedule["tsinghua"]["graph_stub"]
		resp_probs = yield self.run_model(graph, stub, [words])  

		resp_probs = resp_probs[0:10]

		# get posterior probability
		pairs = [(words, each[0]) for each in resp_probs]  
		graph, stub = schedule["postprob"]["graph_stub"]
		post_probs = yield self.run_model(graph, stub, pairs)

		# rerank
		cans = [("".join(each[0]), each[1], post_probs[i]) for i, each in enumerate(resp_probs)]
		results = postprob_rerank(words, cans)

		results = results[0:n] 
		debug_infos = [{"rank":i + 1} for i in range(n)]  
		raise gen.Return((results, debug_infos, DESC["Tsinghua"]))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			multi_results.append({"answer":each, "debug_info":infos[i]})
		return multi_results

class GenerateHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
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
		for name in schedule:
			if name.find("_reverse") == -1:
				generators[name] = schedule[name]["graph_stub"]
			else:
				scorers[name] = schedule[name]["graph_stub"]
		lmda = float(self.get_argument("r", schedule[name].get("lmda", 0.5)))
		a = float(self.get_argument("a", schedule[name].get("alpha", 0.6)))
		b = float(self.get_argument("b", schedule[name].get("beta", 0.1)))
		
		query_utf8 = query.encode("utf-8")
		serverlg.info('[chatbot] [QUERY] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), query_utf8))

		multi_models = [self.run_model(g, stub, [query_utf8], use_seg=True) for g, stub in generators.values()]
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
		pairs = ["%s\t%s" % (query_utf8, " ".join(each)) for each in outputs]
		multi_models = [self.run_model(g, stub, pairs, use_seg=False) for g, stub in scorers.values()]
		multi_model_posteriors = yield multi(multi_models)
		posteriors = [sum(each) for each in zip(*multi_model_posteriors)]
		
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

		raise gen.Return((results, debug_infos, DESC["cvae-generate"]))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			if self.debug == "0":
				multi_results.append({"answer":each})
			else:
				multi_results.append({"answer":each, "debug_info":infos[i]})
		random.shuffle(multi_results)
		return multi_results
