import sys
import os 
import re

from models import util 
import codecs

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string("datafile", None, "data file to process, MUST BE SET")
gflags.DEFINE_string("cols", "0", "colomns to consider in datafile(seperated by '\\t'), seperated by ',' ")
gflags.MarkFlagAsRequired('datafile') 
try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)


name = FLAGS.datafile 
cols = [int(c) for c in re.split(",", FLAGS.cols)]

#name = "data/STC-2/test" 
vocab = {}

basename = os.path.basename(name)
dirname = os.path.dirname(name)

count = 0
with codecs.open(name) as f:
    with codecs.open(os.path.join(dirname, basename+".seg"), "w") as fout:
        for line in f:
            segs = re.split("\t", line.strip())
            seged = []
            if len(segs) != 2:
                continue
            for i, each in enumerate(segs):
                if (each.strip().decode("utf-8") < 2):
                    seged = []
                    print "dep %s" % each
                    break
                if i in cols:
                    words = util.tokenize_word(each.strip())
                    if len(words) == 0:
                        print "Empty after seg", each
                        seged = [] 
                        break

                    #tokens = words 
                    tokens = [] 
                    begin = False
                    token = ""
                    for i, w in enumerate(words):
                        if not begin and w == "[": 
                            token += w 
                            begin = True
                        elif begin and w == "]":
                            token += w 
                            if len(token) <= 10: 
                                token = re.sub(" +", "", token)
                                tokens.append(token)
                            token = ""
                            begin = False
                        elif not begin:
                            token += w 
                            token = re.sub(" +", "", token)
                            tokens.append(token)
                            token = ""
                        elif begin and w != "[":
                            token += w

                    for w in tokens:
                        if w not in vocab:
                            vocab[w] = 0
                        vocab[w] += 1
                    seged.append(" ".join(tokens))
                else:
                    seged.append(each)
            if seged == []:
                continue
            fout.write("%s\n" % ("\t".join(seged)))
            #print "%s\n" % ("\t".join(seged))
            count += 1
            if count % 10000 == 0:
                print "%d segmented..\r" % count,
        print "%d segmented..\r\n" % count,

print "writing vocab..size: %d" % len(vocab)
with codecs.open(os.path.join(dirname, "vocab"), "w") as f:
    for k, v in vocab.items():
        f.write("%s\t%d\n" % (k, v))
