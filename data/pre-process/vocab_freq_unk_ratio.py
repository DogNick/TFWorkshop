#coding=utf-8
import codecs
import sys
import re

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string("dictfile", None, "dict file to process, MUST BE SET")
gflags.DEFINE_string("unkratio", 0.01, "the most unk ratio to tolerate when scaling down dict size (TopN)")
gflags.MarkFlagAsRequired('dictfile') 
try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)

filename = FLAGS.datafile 
unkratio = FLAGS.unkratio

total = 0
word_freqs = [] 
with codecs.open(filename) as f:
	for line in f:
		line = line.strip()
		w, f = re.split("\t", line)
		word_freqs.append((w, int(f)))
		total += int(f)

tail_freq = 0
words_num = 0
for i in range(len(word_freqs)-1, 0, -1):
	tail_freq += word_freqs[i][1]
	tail_ratio = tail_freq * 1.0 / total * 1.0
	print "%d, %.6f%%\r\n" % (i, tail_ratio * 100),
	if tail_ratio >= unkratio:
		words_num = i
		break
