#coding=utf-8
from ModelHandler import *
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
		n = self.get_argument('n', 5)
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[DeepServer] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()

		results = []
		debug_infos = []
		 
		graph_stubs = [schedule[name]["graph_stub"] for name in schedule]
		model_names = [name for name in schedule]
		multi_models = []
		for graph, stub in graph_stubs:
			graph.conf.input_max_len = schedule[name].get("max_in", graph.conf.input_max_len)
			multi_models.append(self.run_model(graph, stub, [query.encode("utf-8")]))
		outs = yield multi(multi_models)
		
		res = []
		for i, out in enumerate(outs):
			outputs, probs, attns = out["outputs"], out["probs"], out["attns"]
			for b in range(1): # batch_is_one for the request
				res.extend(score_with_prob_attn(model_names[i], outputs[b], probs[b], attns[b]))

		sorted_res = sorted(res, key=lambda x:x[3], reverse=True)
		results = [each[1] for each in sorted_res[0:n]] 
		debug_infos = [{"final":str(each[3]), "orig":str(each[4]),
							"lp":str(each[5]), "model":each[0]} 
								for each in sorted_res[0:n]]

		raise gen.Return((results, debug_infos, DESC["generate"]))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			multi_results.append({"answer":each, "debug_info":infos[i]})
		random.shuffle(multi_results)
		return multi_results
