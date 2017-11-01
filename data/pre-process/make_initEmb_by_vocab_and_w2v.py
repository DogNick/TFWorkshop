#coding=utf-8
import numpy as np
import codecs
import math
import sys
import os
import re

import gflags

# Create an npy file given a google word2vec file(not binary) within the given dictfile
# those unknown to dictfile in word2vec will be initialized randomly

FLAGS = gflags.FLAGS
gflags.DEFINE_string("dictfile", None, "dict file that hold all tokens used in model training")
gflags.DEFINE_string("w2vfile", None, "google word2vec file, maybe very large")
gflags.DEFINE_integer("topn", None, "Top N frequency tokens u want to filter the data with")

gflags.MarkFlagAsRequired('dictfile') 
gflags.MarkFlagAsRequired('topn') 
gflags.MarkFlagAsRequired('w2vfile') 

try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)

w2v = {}
dim = 0 
token_cnt = 0

dicfile = FLAGS.dictfile 
w2vfile = FLAGS.w2vfile 
topn = int(FLAGS.topn)
npyfile = os.path.join(os.path.dirname(dicfile), os.path.basename(w2vfile) + ".npy")

with codecs.open(w2vfile) as f: 
    for line in f:
        line = line.strip()
        segs = re.split(" +", line)  
        if len(segs) == 2:
            token_cnt = int(segs[0])
            dim = int(segs[1])
        else:
            w2v[segs[0].strip()] = segs[1:] 

print "word2vec tokens: %d, dim: %d" % (token_cnt, dim)

dic = [] 
with codecs.open(dicfile) as f:
    count = 0
    for line in f:
        line = line.strip()
        dic.append(line)
        count = count + 1
        if count == topn:
            break

embedding = np.zeros((len(dic), dim), dtype = np.float32)

for i, each in enumerate(dic):
    if each in w2v:
        vec = w2v[each]
        for j, f in enumerate(vec):
            embedding[i, j] = float(f)
    else:
        embedding[i] = 2 * math.sqrt(3) * (np.random.rand(dim) - 0.5)
        print ("%s(random)" % each) 

np.save(npyfile, embedding)
