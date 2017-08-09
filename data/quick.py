import codecs
import re
import sys
dicname = sys.argv[1]
filename = sys.argv[2]
voca_size = int(sys.argv[3])

dic = {} 
with codecs.open(dicname) as f:
    for line in f:
        dic[line.strip()] = len(dic) 
        if len(dic) == voca_size: 
            break


with codecs.open(filename) as f:
    with codecs.open(filename + ".quick", "w") as fout:
        count = 0
        for line in f:
            line = line.strip()
            #line = re.sub(" +", " ", line)
            segs = re.split(r"\t", line)
            if len(segs) != 2:
                print "not 2 segs"
                print segs
                print line
                continue
            ps = segs[0].split()
            rs = segs[1].split()
            if len(rs) < 3 or len(rs) > 15: 
                #print "wrong length"
                #print rs
                continue

            #skip = False
            #for each in ps:
            #    if each not in dic:
            #        skip = True
            #        print each
            #        break
            #if skip:
            #    print "skip ps"
            #    print ps
            #    continue

            #for each in rs:
            #    if each not in dic:
            #        skip = True
            #        break
            #if skip:
            #    #print "skip rs"
            #    #print rs
            #    continue
                 
            p, r = segs
            fout.write("%s\t%s\n" % (p.strip(), r.strip()))
            #print "%s\t%s" % (p.strip(), r.strip())
            count += 1
            if count % 100000 == 0:
                print "Processed %d\r" % count,
        print "Processed %d\r\n" % count,
