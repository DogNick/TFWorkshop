import codecs
import sys
import re
import os 


filename = sys.argv[1]
outname = sys.argv[2] 
cols = sys.argv[3]
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
    f.write("_PAD\n")
    f.write("_GO\n")
    f.write("_EOS\n")
    f.write("_UNK\n")
    for each in out:
        f.write("%s\n" % each[0])


