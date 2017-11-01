import codecs
import sys
import re
import os 

import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_string("datafile", None, "datafile to process")
gflags.DEFINE_string("outname", None, "Vocab name out")
gflags.DEFINE_string("cols", None, "colomns to be considered in data, seperated by ','")

gflags.MarkFlagAsRequired('datafile') 
gflags.MarkFlagAsRequired('outname') 
gflags.MarkFlagAsRequired('cols')
try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)

filename = FLAGS.datafile 
outname = FLAGS.topn 
cols = FLAGS.cols 

cols = [int(c) for c in re.split(",", cols)]
vocab = {}
with codecs.open(filename) as f:
    for line in f: 
        segs = re.split("\t", line.strip())
        for i, seg in enumerate(segs):
            if i not in cols: 
                continue
            if seg == "</s>":
                continue
            for w in seg.split():
                if w not in vocab:
                    vocab[w] = 0
                vocab[w] += 1

out = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
with codecs.open(outname, "w") as f:
	with codecs.open(outname + ".freq", "w") as f2:
		f.write("_PAD\n")
		f.write("_GO\n")
		f.write("_EOS\n")
		f.write("_UNK\n")
		for each in out:
			f.write("%s\n" % each[0])
			f2.write("%s\t%d\n" % (each[0], each[1]))
