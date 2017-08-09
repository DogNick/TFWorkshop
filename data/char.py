#coding=utf-8
import sys
import re

cols = re.split(",", sys.argv[1])
cols = [int(c) for c in cols]

for line in sys.stdin:
    line = line.strip()
    segs = re.split("\t+", line)

    newsegs = []
    for i, seg in enumerate(segs):
        if i in cols: 
            chars = []
            words = seg.split()
            for w in words:
                for c in w.decode("utf-8"):
                    chars.append(c)
            newsegs.append(" ".join(chars))
        else:
            newsegs.append(seg)
    if newsegs[-1] == "11.0":
        newsegs[-1] = "1.0"

    print "\t".join(newsegs).encode("utf-8")
            

            
