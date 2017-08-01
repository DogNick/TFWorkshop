#coding=utf-8
from wordseg_python import Global

PUNC = ['，', ',', '。', '.', '！', '!', '"', '?', '？']
COMMON = ['你', '我', '的', '哈', '他', '她']
        
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


def work(sent):
    before = split(sent)
    after = remove_rep(before)
    if before != after:
        print('before: '+' '.join(before))
        print(' after: '+' '.join(after))

#with open('0330.txt') as f:
#    for line in f:
#        work(line.strip().split()[1])
