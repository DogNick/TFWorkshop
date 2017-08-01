#coding=utf-8
import codecs
import sys
import re
from util import tokenize_word

def overlap(s1, s2):
    n1, n2 = len(s1), len(s2)
    dp = [[0 for __ in range(n2+1)] for _ in range(n1+1)]
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            dp[i][j] = max(dp[i][j-1], dp[i-1][j])
            if s1[i-1] == s2[j-1]:
                dp[i][j] = max(dp[i][j], dp[i-1][j-1]+1)
    return dp[n1][n2]


res = {} 
with codecs.open(sys.argv[1], "r") as f:
    for line in f:
        line = line.strip()
        p, r, _ = re.split("\t", line)
        if p not in res:
            res[p] = []
        words, pos = tokenize_word(r)
        res[p].append(words)

for p in res:
    selected = []
    for i, resp in enumerate(res[p]):
        isdiff = True 
        for j in range(i):
            if overlap(res[p][j], res[p][i]) >= 4:
                isdiff = False
                break
        if isdiff:
            selected.append("".join(resp))
    if len(selected) < 10:
        print "resps less than 10: %s" % p
    for resp in selected[0:10]:
        print "%s\t%s" % (p, resp)





