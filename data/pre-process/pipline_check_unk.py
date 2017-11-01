import codecs
import re
import sys

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string("dictfile", None, "dict file")
gflags.DEFINE_integer("topn", None, "Top N frequency tokens u want to filter the data with")

# Given a dict file and a topN number, print sentence input by pipline with unknown tokens tailed by '_UNK'
#
gflags.MarkFlagAsRequired('dictfile') 
gflags.MarkFlagAsRequired('topn') 

try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)



with codecs.open(FLAGS.dictfile) as f:
	dic = {f.next().strip():1 for n in range(int(FLAGS.topn))}

for line in sys.stdin:
	line = line.strip()
	words = re.split(" +", line)	
	checked = [] 
	hasUNK = False 
	for w in words:
		if w not in dic:
			w = "%s_UNK" % w
			hasUNK = True
		checked.append(w)
	if hasUNK:
		print " ".join(checked)
