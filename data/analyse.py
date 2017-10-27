import sys
import re
import gflags
from time import strftime
import time

gflags.DEFINE_string("logtype", "ANALYSE", "ANALYSE/BOT/USR/USR_BOT")
gflags.DEFINE_string("key", "uid", "sort by gid or uid")
gflags.DEFINE_string("interval", "-1", "seged by time interval")

FLAGS=gflags.FLAGS
FLAGS(sys.argv)

class MatchFields(object):
	def __init__(self, logtype):
		if logtype == "ANALYSE":
			self._reg_fields = re.compile("\[(#ANALYSE\-VERBOSE#)\]\[UUID:(.*?)\]\[timestamp:(.*?)\]\[time:(.*?)\]\[groupid=(.*?)\]\[uid=(.*?)\]\[query=(.*?)\]\[data.*?\[answer=(.*?)\]\[from=(.*?)\]")
		elif logtype == "BOT":
			self._reg_fields = re.compile("\[(#BOT#)\]\[UUID:(.*?)\]\[timestamp:(.*?)\]\[groupid=(.*?)\]\[uid=(.*?)\]\[answer=(.*?)\]\[from=(.*?)\]")
		elif logtype == "USR":
			self._reg_fields = re.compile("\[(#USR#)\]\[UUID:(.*?)\]\[timestamp:(.*?)\]\[groupid=(.*?)\]\[uid=(.*?)\]\[query=(.*?)\]\[data")
		elif logtype == "USR_BOT":
			self._reg_fields = re.compile("\[(#BOT#|#USR#)\]\[UUID:(.*?)\]\[timestamp:(.*?)\]\[groupid=(.*?)\]\[uid=(.*?)\]\[(query|answer)=(.*?)\](\[data.*|\[from=(.*?)\])")
		else:
			print "unknown logtype"
			exit(0)
		self._logtype = logtype
		self.buf = {} 
		self.gmap = {}
		self.umap = {}

	def match(self, record_str):
		matched = self._reg_fields.search(record_str)
		if matched:
			if self._logtype == "ANALYSE":
				(logtype, uuid, ts, t, gid, 
				uid, query, ans, fr) = matched.group(1), matched.group(2), matched.group(3), matched.group(4), matched.group(5), matched.group(6), matched.group(7), matched.group(8), matched.group(9)
			elif self._logtype == "BOT":
				(logtype, uuid, ts, t, gid,
				uid, query, ans, fr) = matched.group(1), matched.group(2), matched.group(3), None, matched.group(4), matched.group(5), None, matched.group(6), matched.group(7)
			elif self._logtype == "USR":
				(logtype, uuid, ts, t, gid,
				uid, query, ans, fr) = matched.group(1), matched.group(2), matched.group(3), None, matched.group(4), matched.group(5), matched.group(6), None, None
			elif self._logtype == "USR_BOT":
				if matched.group(1) == "#USR#":
					(logtype, uuid, ts, gid,
					uid, query, ans, fr) = matched.group(1), matched.group(2), matched.group(3), matched.group(4), matched.group(5), matched.group(7), None, None
				else:
					(logtype, uuid, ts, gid,
					uid, query, ans, fr) = matched.group(1), matched.group(2), matched.group(3), matched.group(4), matched.group(5), None, matched.group(7), matched.group(9)
			else:
				return None
			query = re.sub("[\t ]+", " ", str(query))
			ans = re.sub("[\t ]+", " ", str(ans))
			return (logtype, query, ans, fr, gid, uid, ts) 
		else:
			return None

	def read(self):
		for each in sys.stdin:
			each = each.strip()
			res_tuple = self.match(each)
			if res_tuple:
				if res_tuple[-3] not in self.gmap: 
					self.gmap[res_tuple[-3]] = "#GRP_%d#" % len(self.gmap)
				if res_tuple[-2] not in self.umap:
					self.umap[res_tuple[-2]] = "#USR_%d#" % len(self.umap)
				key = self.umap[res_tuple[-2]] if FLAGS.key == "uid" else self.gmap[res_tuple[-3]]
				if key not in self.buf:
					self.buf[key] = [] 
				self.buf[key].append(res_tuple)

	def out(self):
		interval = float(FLAGS.interval)
		for uid, v in self.buf.items():
			v = sorted(v, key=lambda x:float(x[-1]))
			for i, utter in enumerate(v):
				if interval > 0 and i > 0 and float(utter[-1]) - float(v[i-1][-1]) >= interval:
					print ""
				print self.format(utter)
			print ""

	def format(self, res_tuple):
		# must be (logtype, q, ans, fr, gid, uid, ts)
		if res_tuple[0] == "#ANALYSE-VERBOSE#":
			return "%s\t%s\t%s\t%s" % (res_tuple[1], res_tuple[2], res_tuple[3], strftime("%Y-%m-%d_%H:%M:%S", time.localtime(float(res_tuple[-1]))))
		elif res_tuple[0] == "#BOT#":
			return "%s\t%s\t%s\t%s\t%s" % (self.gmap[res_tuple[-3]], strftime("%Y-%m-%d_%H:%M:%S", time.localtime(float(res_tuple[-1]))), res_tuple[0], res_tuple[2], res_tuple[3])
		elif res_tuple[0] == "#USR#":
			return "%s\t%s\t%s\t%s\t%s" % (self.gmap[res_tuple[-3]], strftime("%Y-%m-%d_%H:%M:%S", time.localtime(float(res_tuple[-1]))), self.umap[res_tuple[-2]], res_tuple[1], res_tuple[3])

def main():
	mf = MatchFields(FLAGS.logtype)
	mf.read()
	mf.out()

if __name__ == "__main__":
	main()
