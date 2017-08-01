#coding=utf-8
from ModelHandler import *
class DeepMatchHandler(ModelHandler):
	def handle(self, plan_name="default"):  
		# collect infos for current request
		query = self.get_argument('query', None)
		cans = self.get_argument('cans', None)
		# check valid
		if not query:
			ret = {}
			ret["status"] = "missing params"
			serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
			self.write(json.dumps(ret, ensure_ascii=False))
			self.finish()
			return

		records = []
		query_utf8 = query.encode("utf-8")
		cans = json.loads(cans) 
		for each in cans:
			records.append("%s\t%s\t%s" % (query_utf8, each[0].encode("utf-8"), each[1].encode("utf-8")))

		scores = []
		if plan_name == "default":
			scores = yield self.run_model("stc-2-interact", records)  
		raise gen.Return(scores)

	def form_multi_results(self, model_name, model_out): 
		conf = confs[model_name]
		debug_info = {"name":model_name, "server":conf.tf_server,"model_kind":conf.model_kind}
		multi_results = [{"scores":[str(s) for s in model_out], "debug_info":debug_info}]
		return multi_results 
