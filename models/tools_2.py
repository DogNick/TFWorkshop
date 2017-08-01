#coding=utf-8
from wordseg_python import Global

PUNC = ['，', ',', '。', '.', '！', '!', '"', '?', '？']
COMMON = ['你', '我', '的', '哈', '他', '她', '是', '好', '了', '啊']
POST = 'final_res'
RESP = [('response_v1_final', 45, 65), ('response_v2_final', 30, 40)]


def split(sent):
    sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
    tuples = [(word.decode("gbk").encode("utf-8"), pos)
            for word, pos in Global.GetTokenPos(sent)]
    return [each[0] for each in tuples]

def remove_rep(sent):
    length = 4
    while length > 1:
        index = -1
        for i in range(len(sent)-2*length+1):
            if ''.join(sent[i:i+length]) == ''.join(sent[i+length:i+2*length]):
                index = i
                break
        if index >= 0:
            #print (length, index)
            sent = sent[:i]+sent[i+length:]
        else:
            length -= 1
    length = 6
    while length > 0:
        index = -1
        for i in range(len(sent)-2*length):
            if (sent[i+length] in PUNC):
                if ''.join(sent[i:i+length]) == ''.join(sent[i+1+length:i+1+2*length]):
                    index = i
                    break
        if index >= 0:
            #print (length, index)
            sent = sent[:i]+sent[i+length+1:]
        else:
            length -= 1
    return sent

def check_length(resp):
    if len(resp) <= 3:
        return False
    if len(resp) >= 12:
        return False
    return True

def check_overwritten(resp):
    charset = {}
    for item in resp:
        charset[item] = charset.get(item, 0) + 1
    for key, value in charset.items():
        if key in PUNC:
            if value > 3:
                return False
        elif key in COMMON:
            if value > 2:
                return False
        else:
            if value > 1:
                return False
    return True

def check_pattern(resp):
    r = ''.join(resp)
    if (r.find('我也') >= 0) and (len(resp) <= 8):
        return False
    if ((r.find('谢谢') >= 0) or (r.find('感谢') >= 0)) and\
            ((r.find('鼓励') >= 0) or (r.find('支持') >= 0)):
        return False
    return True

def overlap(s1, s2):
    n1, n2 = len(s1), len(s2)
    dp = [[0 for __ in range(n2+1)] for _ in range(n1+1)]
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            dp[i][j] = max(dp[i][j-1], dp[i-1][j])
            if s1[i-1] == s2[j-1]:
                dp[i][j] = max(dp[i][j], dp[i-1][j-1]+1)
    return dp[n1][n2]

def work(post, responses):
    candidates = []
    for p, r, i in responses:
        if check_length(r) and check_overwritten(r):
            candidates.append((p, r, i))
    candidates.sort(key=lambda x:x[0])
    result = [candidates[0]]
    for now in candidates[1:]:
        p = [overlap(x[1], now[1]) for x in result]
        if max(p) <= 3:
            result.append(now)
        if len(result) > 10: break
    while len(result) < 10:
        result = result + [result[-1]]
    return result[:10]

post = []
with open('../%s' % POST) as f:
    for line in f:
        post.append(line.strip())

resp = [[] for _ in post]
for path, st, ed in RESP:
    for i in range(st, ed):
        with open('seq2seq/%s/%d.txt' % (path, i)) as f:
            index = -1
            for line in f:
                if line[0:3] == '###':
                    index += 1
                else:
                    tokens = line.strip().split()
                    resp[index].append((float(tokens[0]), tokens[1:], i))

with open('result.txt', 'w') as f:
    for p, r in zip(post, resp):
        results = work(p, r)
        for reply in results:
            f.writelines('%s\t%s\n' % (p, ''.join(reply[1])))
