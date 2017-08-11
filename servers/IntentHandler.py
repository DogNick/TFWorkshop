from ModelHandler import *
class IntentHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  

		query = self.get_argument('query', None)
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[Intent] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()
		results = []
		debug_infos = []
		 
		graph_stubs = [schedule[name]["graph_stub"] for name in schedule]
		model_names = [name for name in schedule]
		# Multi model compatible, but here just one model exists
		multi_models = []
		for graph, stub in graph_stubs:
			multi_models.append(self.run_model(graph, stub, [query.encode("utf-8")]))
		outs = yield multi(multi_models)
		# only use one model
		res = {} 
		for k,v in outs[0].items():
			if isinstance(v, list):
				res[k] = str(v[0])
			else:
				res[k] = str(v)
				
		raise gen.Return(([res], [], "tianchuan_cnn"))	

	def form_multi_results(self, plan_results, infos): 
		serverlg.info(plan_results)
		return plan_results 
