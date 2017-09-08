#coding=utf-8
from ModelHandler import *
import scipy.spatial
def cos_cdist(matrix, vector):
	"""
	    Compute the cosine distances between each row of matrix and vector.
	"""
	v = vector.reshape(1, -1)
	return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

class ScoreHandler(ModelHandler):
	@tornado.gen.coroutine
	def handle(self):  
		# collect infos for current request
		data = json.loads(self.request.body)
		query = data["query"]
		cans = data["cans"]
		#query = self.get_argument('query', None)
		#cans = self.get_argument('cans', None)
		# check valid
		#if not query:
		#	ret = {}
		#	ret["status"] = "missing params"
		#	serverlg.info('[chatbot] [ERROR: missing params] [REQUEST] [%s] [%s]' % (time.strftime('%Y-%m-%d %H:%M:%S'), self.request.uri))
		#	self.write(json.dumps(ret, ensure_ascii=False))
		#	self.finish()
		#	return

		# this is for query embedding
		scorers = {}
		for name in schedule:
			scorers[name] = schedule[name]["graph_stub"]

		records = [query.encode("utf-8")]
		for each in cans:
			records.append(each.encode("utf-8"))

		# use multi-model result
		multi_models = [self.run_model(g, stub, records, use_seg=False) for g, stub in scorers.values()]
		outs = yield multi(multi_models)

		# only one model
		one_model_out = outs[0]
		results = [outs[0][i] for i in range(1, len(outs[0]))] 
		debug_infos = [{} for each in results]

		raise gen.Return((results, debug_infos, "score by rnn_enc and sum of word embs"))

	def form_multi_results(self, plan_results, infos): 
		return plan_results
		#multi_results = []
		#for i, each enumerate(plan_results):
		#	multi_results.append(
		#conf = confs[model_name]
		#debug_info = {"name":model_name, "server":conf.tf_server,"model_kind":conf.model_kind}
		#multi_results = [{"scores":[str(s) for s in model_out], "debug_info":debug_info}]
		#return multi_results 
