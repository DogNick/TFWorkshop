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
		posts = data["cans"]
		resps = data["responses"]
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

		# prepare data records type for different model
		query_clean_joint = util.clean_en(sentence=query.encode("utf-8"), need_pos=False, joint_prime=True)
		query_clean = util.clean_en(sentence=query.encode("utf-8"), need_pos=False, joint_prime=True)
		posterior_records =["%s\t%s" % (query_clean_joint, util.clean_en(sentence=each.encode("utf-8"), need_pos=False, joint_prime=True)) for each in resps]
		similarity_records = [query_clean]
		for each in posts:
			similarity_records.append(util.clean_en(sentence=each.encode("utf-8"), need_pos=False, joint_prime=False))

		# different records type for different models
		multi_models = []
		for name in schedule:
			g, stub = schedule[name]["graph_stub"]
			if name.find("_reverse") != -1:
				multi_models.append(self.run_model(g, stub, posterior_records, use_seg=False))
			else:
				multi_models.append(self.run_model(g, stub, similarity_records, use_seg=False))
		# model in parallell and async
		outs = yield multi(multi_models)
	
		results = [] 
		for i in range(len(outs[0])):
			ensembled = {}
			for each_model_res in outs:
				for k,v in each_model_res[i].items():
					ensembled[k] = v
			print ensembled
			results.append(ensembled)
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
