import codecs
import sys
import re

name = sys.argv[1]
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

        
