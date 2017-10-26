#coding=utf-8
import numpy as np
import codecs
import math
import sys
import os
import re


w2v = {}
dim = 0 
token_cnt = 0

dicfile = sys.argv[1]
w2vfile = sys.argv[2] 
Nmax = int(sys.argv[3])
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
        if count == Nmax:
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
