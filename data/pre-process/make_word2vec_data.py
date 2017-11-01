import codecs
import sys
import re

import gflags
FLAGS = gflags.FLAGS

# transfer datafile with multi colomns into datafile with one colomn a line for google word2vec corpus

gflags.DEFINE_string("datafile", None, "data file to make from")
gflags.MarkFlagAsRequired('datafile') 

try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)

name = FLAGS.datafile 
with codecs.open(name) as f:
    with codecs.open(name + ".forw2v", "w") as fout:
        last = ""
        count = 0
        for line in f:
            segs = re.split("\t", line.strip())
            if last != "" and segs[0] != last: 
                fout.write("%s\n" % last)
            last = segs[0]
            fout.write("%s\n" % segs[1])
            count += 1
            if count % 100000 == 0:
                print "%d processed\r" % count,
        fout.write("%s\n" % last)
        print "%d processed\r\n" % count,

        
