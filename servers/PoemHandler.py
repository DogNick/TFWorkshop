#coding=utf-8
import codecs
from ModelHandler import *
# Global whitlist for poem servers
match_poem_whitelist = {} 
with codecs.open("server_data/match_poem.whitelist", "r", "utf-8") as f:
	for line in f:
		p, r_str = re.split("\t", line.strip())
		match_poem_whitelist[p] = re.split(" +", r_str)	

class MatchPoemHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
		query = self.get_argument('query', None)
		n = int(self.get_argument('n', 10))
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()

		# remove whitelist item from models
		results = []
		infos = []

		if query in match_poem_whitelist:
			results.extend([("whitelist", each.encode("utf-8"), 0.0) for each in match_poem_whitelist[query]])
		else:
			graph_stubs = [schedule[name]["graph_stub"] for name in schedule]
			model_names = [name for name in schedule]
			multi_models = []
			for graph, stub in graph_stubs:
				#graph.conf.input_max_len = schedule[name].get("max_in", graph.conf.input_max_len)
				seged_query = " ".join([w for w in query]).encode("utf-8")
				serverlg.info("%s %d" % (seged_query, len(query)))
				multi_models.append(self.run_model(graph, stub, [seged_query], use_seg=False))
			outs = yield multi(multi_models)
			# fused all model results of one example (because currently batch_size = 1)
			for i, out in enumerate(outs):
				outputs, probs, attns = out["outputs"], output["probs"], output["attn"]
				match_sens = ["".join(each) for each in outputs[0]]
				match_sen_cans, scores = score_couplet(model_names[i], query.encode("utf-8"), match_sens, probs[0]) 
				results.extend([(model_names[i], match_sen_cans[k], scores[k]) for k in range(len(scores))])

		sorted_res = sorted(results, key=lambda x:x[2], reverse=True)

		# select different begining
		selected = []	
		infos = []
		for each in sorted_res:
			if each[2] < -100:
				continue
			selected.append(each[1])
			infos.append({"model_name":each[0], "prob":str(each[2])})
			print each[0], each[1], each[2]
			if len(selected) == n:
				break

		raise gen.Return((selected, infos, DESC["matchpoem"]))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			multi_results.append({"answer":each, "debug_info":infos[i]})
		random.shuffle(multi_results[0:3])
		return multi_results

class JudgePoemHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
		query = self.get_argument('query', None)
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()
			return

		examples = []
		segs = re.split(u"[，！？。 ]+", query)
		serverlg.info(len(segs))

		for seg in segs:
			if len(seg) == 5 or len(seg) == 7 or len(seg) == 4: 
				examples.append(seg)

		if examples == []:
			raise gen.Return(([], [], DESC["judgepoem"]))

		# remove whitelist item from models
		examples_to_infer = []
		results = []
		infos = []
		for each in examples:
			if each in match_poem_whitelist:
				results.append(each)
				infos.append({"judge_prob":"1.0"})
				serverlg.info("whitelist %s" % each)
			else:
				examples_to_infer.append((" ".join([c for c in seg])).encode("utf-8"))
				serverlg.info("infer %s" % examples_to_infer[-1])
				
		if examples_to_infer:
			graph_stubs = [schedule[name]["graph_stub"] for name in schedule]
			model_names = [name for name in schedule]
			multi_models = []
			for graph, stub in graph_stubs:
				multi_models.append(self.run_model(graph, stub, examples_to_infer, use_seg=False))
			tag_probs = yield multi(multi_models)
			for i, each in enumerate(examples):
				results.append(re.sub(" +", "", each))
				infos.append({"judge_prob": str(tag_probs[i])})

		raise gen.Return((results, infos, DESC["judgepoem"]))

	def form_multi_results(self, plan_results, infos): 
		multi_results = []
		for i, each in enumerate(plan_results):
			serverlg.info('[DeepServer] [ans] %s [info] %s' % (each, str(infos[i])))
			multi_results.append({"answer":each, "debug_info":infos[i]})
		random.shuffle(multi_results)

		return multi_results
