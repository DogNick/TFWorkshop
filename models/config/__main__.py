from __init__ import *
import re
import sys
if len(sys.argv) == 1:
	print "======================================================"
	print "==========          Configurations      =============="
	print "======================================================"
	keys = sorted(confs.keys())
	for each in keys:
		print "====   %s " % each
else:
	reg = sys.argv[1]
	keys = sorted(confs.keys())
	for each in keys:
		if re.search(reg, each) != None:
			print "======== %s " % each
			for k, v in confs[each].__dict__.items():
				print "=== %s ----- %s" % (k, str(v))
			print "======================================================"
			print "======================================================"
			print ""
