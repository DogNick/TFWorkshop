import codecs
import sys
import os
import re

filename = sys.argv[1]
distdir = sys.argv[2]
key_seg = int(sys.argv[3])

data = []
key_count = {}
count = 0

with codecs.open(filename) as f:
    for line in f:
        line = line.strip()
        p, r = re.split("\t", line)[0:2]
        
        r_words = re.split(" +", r)
        key = " ".join(r_words[:key_seg])
        if key not in key_count:
            key_count[key] = 0
        key_count[key] += 1
        data.append((line, key))
        count += 1
        if count % 100000 == 0:
            print "Read %d\r" % count,
    print "Read %d\r" % count,

name = os.path.basename(filename)

#for key, count in sorted(key_count.items(), key=lambda x:x[1], reverse=True):
#    print key, count
with codecs.open(os.path.join(distdir, name) + ".downsample", "w") as f:
    for line, key in data:
        wgts = 1.0
        if key_count[key] > 100: 
            wgts = 100.0 / float(key_count[key])
        f.write("%s\t%.5f\n" % (line, wgts))
