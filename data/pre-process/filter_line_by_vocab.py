#coding=utf-8
import sys
import codecs
import re

sorteddicfile = sys.argv[1]
topN = int(sys.argv[2])

dic = {}
with codecs.open(sorteddicfile, "r") as f:
	dic = {f.next().strip():1 for i in range(topN)}		

for line in sys.stdin:
	line = line.strip()
	skip = False
	for w in re.split("[\t ]+", line):
		if w not in dic:
			skip = True
			break
	if not skip:
		print line
