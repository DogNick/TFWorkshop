import codecs
import re
import sys

with codecs.open(sys.argv[1]) as f:
	dic = {f.next().strip():1 for n in range(int(sys.argv[2]))}

for line in sys.stdin:
	line = line.strip()
	words = re.split(" +", line)	
	checked = [] 
	hasUNK = False 
	for w in words:
		if w not in dic:
			w = "%s_UNK" % w
			hasUNK = True
		checked.append(w)
	if hasUNK:
		print " ".join(checked)
