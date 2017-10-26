import codecs
import random
import sys
import os
import re
data = [] 
idx = []
name = sys.argv[1]
distdir = sys.argv[2]
with codecs.open(name) as f:
    last_q = ""
    begin = 0
    end = 0
    print ("")
    for line in f:
        segs = re.split("\t", line.strip())
        if len(segs) != 2:
            print "not 2", line
            continue
        if segs[0].strip() == "" or segs[1].strip() == "":
            print "empty", line 
            continue
        data.append([segs[0], segs[1]])
        if last_q != segs[0] and last_q != "":
            idx.append(end)
        last_q = segs[0]
        end += 1
        if end % 200000 == 0:
            print "%d read\r" % end,
    print "%d read\r\n" % end,

print "posts: %d\n" % len(idx)

name = os.path.basename(name)
N = 2 
fout = codecs.open(os.path.join(distdir, name+".neg"), "w")
for i in range(len(idx)):
    if i == 0:
        begin = 0
    else:
        begin = idx[i-1]

    if i % 10000 == 0:
        print "sampled posts: %d\r" % i,
    end = idx[i]
    m = begin
    n = len(data) - end 

    for j in range(begin, end):
        if random.random() > m * 1.0 / ((m + n) * 1.0):
            q_idx = random.sample(xrange(end, len(data)), N)
        else:
            q_idx = random.sample(xrange(0, begin), N)

        if random.random() > m * 1.0 / ((m + n) * 1.0):
            r_idx = random.sample(xrange(end, len(data)), N)
        else:
            r_idx = random.sample(xrange(0, begin), N)

        # q, q r, r-
        for r_i in r_idx:
            q, p, r, neg_r = data[j][0].strip(), data[j][0].strip(), data[j][1].strip(), data[r_i][1] 
            if q == "" or p == "" or r == "" or neg_r == "":
                print "DEP:", q, p, r, neg_r
                continue
            fout.write("%s\n" % "\t".join([q, p, r, neg_r]))

        # q, q* r, r-
        for q_i, r_i in q_idx, r_idx:
            q, p, r, neg_r = data[j][0].strip(), data[q_i][0].strip(), data[j][1].strip(), data[r_i][1] 
            if q == "" or p == "" or r == "" or neg_r == "":
                print "DEP:", q, p, r, neg_r
                continue
            fout.write("%s\n" % "\t".join([q, p, r, neg_r]))
        # q, q* r, r*-
        for q_i in q_idx:
            q, p, r, neg_r = data[j][0].strip(), data[q_i][0].strip(), data[j][1].strip(), data[q_i][1]
            if q == "" or p == "" or r == "" or neg_r == "":
                print "DEP:", q, p, r, neg_r
                continue
            fout.write("%s\n" % "\t".join([q, p, r, neg_r]))
