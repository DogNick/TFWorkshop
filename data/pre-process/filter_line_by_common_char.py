#coding=utf-8
import codecs
import re
import sys
filename = sys.argv[1]
with codecs.open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if re.search(u"[^\u4e00-\u9fa5，。：？！0-9.０１２３４５６７８９a-zA-Z:?!\"\- \t]", line.decode("utf-8")) != None:
            continue
        print line
