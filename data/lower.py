import codecs
import re
filename = "pair.filter1"
with codecs.open(filename) as f:
    for line in f:
        line = line.strip()
        line = re.sub(" +", " ", line)
        segs = re.split("\-\-\-\-\-", line)
        if len(segs) != 2:
            continue
        p, r = segs 
        if (len(p.split()) < 2 or len(r.split()) < 2):
            continue
        p = eval(p).lower().strip()
        r = eval(r).lower().srrip()
        #print p.strip(), r.strip()
        print "%s\t%s" % (p, r)
