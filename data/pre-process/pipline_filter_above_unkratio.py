import codecs
import sys
import re

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string("dictfile", None, "dict file with frequencies and reverse-sorted")
gflags.DEFINE_integer("topn", None, "Top N frequency tokens u want to filter the data with")
gflags.DEFINE_float("unkratio", 0.2, "unk token ratio limit for each sentence to filter")

gflags.MarkFlagAsRequired('dictfile') 
gflags.MarkFlagAsRequired('topn') 

try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)


dictname = FLAGS.dictfile 
topn = FLAGS.topn 
unkratio = FLAGS.unkratio 

dic = {}
count = 0
with codecs.open(dictname) as f:
	for line in f:
		line = line.strip()
		dic[re.split("\t", line)[0]] = 0
		count += 1
		if count == topn:
			break

for line in sys.stdin:
	line = line.strip()
	segs = re.split("\t", line)
	if len(segs) != 2:
		continue
	ps = re.split(" +", segs[0]) 
	rs = re.split(" +", segs[1]) 
	unkcount = 0
	for w in ps:
		if w not in dic:
			unkcount += 1
	if unkcount * 1.0 / len(ps) * 1.0 > unkratio: 
		continue
	unkcount = 0
	for w in rs: 
		if w not in dic:
			unkcount += 1
	if unkcount * 1.0 / len(rs) * 1.0 > unkratio: 
		continue
	print line
