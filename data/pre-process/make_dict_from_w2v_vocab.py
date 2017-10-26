import codecs
import sys
import re
import os 


filename = sys.argv[1]
outname = sys.argv[2] 
vocab = [
    "_PAD",
    "_GO",
    "_EOS",
    "_UNK"
]
with codecs.open(filename) as f:
    for line in f: 
        segs = re.split("[\t ]+", line.strip())
        if segs[0] == "</s>":
            continue
        vocab.append(segs[0].strip())

with codecs.open(outname, "w") as f:
    for each in vocab:
        f.write("%s\n" % each)
        



