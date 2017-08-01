#!/usr/bin/env python
# coding:gbk
'''
Created on 2013-5-16

@author: felicialin
'''

import sys
import re
# Added by Angel
import os
# End

# we assume data lies in the same dir as this file
curr_dir = os.path.dirname(os.path.abspath(__file__))
# Modified by Angel
if sys.maxint == 9223372036854775807:
    sys.path.append("./TCWordSeg/64")
    sys.path.append("./liblinear/64")
else:
    sys.path.append("./TCWordSeg/32")
    sys.path.append("./liblinear/32")
# End
   
from TCWordSeg import *

GLOBAL_SEG_MODE = TC_U2L | TC_GU | TC_POS | TC_S2D | TC_T2S | TC_CN | TC_PGU | TC_LGU | TC_SGU | TC_CONV | TC_WGU
# Modified by Angel
#TCInitSeg("wordseg_python/dict")
TCInitSeg(curr_dir + "/dict")
# End
seghandle = TCCreateSegHandle(GLOBAL_SEG_MODE)

#stop_words = ["��", "��", "��"]  #ͣ�ô�����
stop_words = []
GLOBAL_SEGWORD = " "  #NԪ����ָ���

'''
����word��token����һ������Ϊһ��token��һ��Ӣ�ĵ���Ϊһ��token���ո񲻼��㣬�����ż������
'''
def GetWordLen(word):
    word = word.decode("gbk", "ignore")
    context = re.compile("([A-Za-z0-9]+)")
    tmp = context.findall(word)        
    count = len(tmp)    
    word = context.sub("", word)    
    for wd in word:
        wd = wd.strip()
        if len(wd) > 0:
            count += 1
    return count 

        
#�ִ�
def GetTokenPos(wd):
    TCSegment(seghandle, wd)
    rescount = TCGetResultCnt(seghandle)
    SegArray = []
#   SegArray = [("<S>",0),]
    for i in range(rescount):
        wordpos = TCGetAt(seghandle, i)     
        bcw = wordpos.bcw
        pos = wordpos.pos
        word = wordpos.word
        '''uword = word.decode("gbk", "ignore")
        if pos == 34 or word in stop_words: #�����Ż���ͣ�ô�ֱ���ӵ�
            continue    
        word_len = GetWordLen(word)
        if word_len > 4 or len(uword) > 6: #���ȹ�����ֱ���ӵ���ͳ��
            continue'''   
        #SegArray += [(wordpos.word, wordpos.pos), ]
        '''if bcw:
            res = word
        else:'''
        #res = '|||' if pos == 34 else word
        #if pos == 34 or word in stop_words:
        #    continue
        SegArray.append((word, pos))
	#SegArray.append(word)
#        SegArray += [("<E>",0)]
    return SegArray 
#������ʻ��ߴ�����߿��ﻰ��
meanless_words = [ "hi", "hello", "the", "nihao", "���", "�Ǻ�", "�պ�", "����", "����", "лл", "��л", "��", "��", "��"
                "��", "��", "��", "ѽ", "��", "��", "��", "��", "��", "��", "��", "Ŷ", "��", "�", "��", "��", "��",
                "��", "��", "��", "��", "��", "��", "��", "��", "��", "����", "ô", "ôô", "��", "ι", "ŶŶ",
                "��ѽ", "���", "��ѽ", "��", "de", "��", "��", "�ٺ�", "����",  "��", "��", "��",
                "��", "��", "û", "��", "�ź�", "�ǰ�", "��", "��", "ɶ", "�", "����", "��", "��ѽ",
                "�¸�", "����", "Ŷ", "���", "����", "߹", "��", "����", "����", "����", "��", "��", "��", "��",
                "����", "����", "�ҿ�", "�ȿ�", "����", "��", "�а�", "����", "��", "ô", "�Ű�", "��ѽ", "��ô", "����",
                "����", "��", "����", "���", "��", "ιι", "����", "Ү", "��Ŷ", "����", "��", "��", "��", "��",
                "�պ�", "��", "�õ�", "����", "Ŷ��", "�ߺ�", "��Ŷ", "զ��", "��", "����", "Ӵ", "ɶ��", "��Ŷ", "��",
                "��", "��", "��", "��", "��", "��", "��", "Ŷ", "��", "����", "����", "��",  "�ٺ�", "����",
                "����", "����", "�ݰ�", "�ǰ�", "ɵ��", "�ǵ�", "�õ�", "�ټ�", "��", "�ð�", "лл", "��",
                
               ]
chengdu_words = ["��", "��", "��", "��", "̫", "Խ", "ͦ", "��","��", 
                 "����", "��", "��", "�ǳ�","����", "Խ��Խ", "Խ��", "��֮","�ܹ�",
                "��", "��", "��", "����", "����", "����", "ȫ��", "����",
                 "����", "����","û����", "û��",]
