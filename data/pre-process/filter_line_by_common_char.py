#coding=utf-8
import codecs
import re
import sys

import gflags

# Filter out sentences with tokens not match regex bellow

FLAGS = gflags.FLAGS
gflags.DEFINE_string("datafile", None, "datafile")
gflags.MarkFlagAsRequired('datafile') 
try:
	FLAGS(sys.argv)
except gflags.FlagsError as e:
	print "\n%s" % e 
	print FLAGS.GetHelp(include_special_flags=False) 
	sys.exit(1)


filename = FLAGS.datafile 

with codecs.open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if re.search(u"[^\u4e00-\u9fa5，。：？！0-9.０１２３４５６７８９a-zA-Z:?!\"\- \t]", line.decode("utf-8")) != None:
            continue
        print line
