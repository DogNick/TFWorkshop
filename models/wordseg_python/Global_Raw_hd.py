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
TCInitSeg("./wordseg_python/wordseg_python/dict")
# End
seghandle = TCCreateSegHandle(GLOBAL_SEG_MODE)

#stop_words = ["的", "是", "啊"]  #停用词数组
stop_words = []
GLOBAL_SEGWORD = " "  #N元输出分隔符

'''
计算word的token数，一个汉字为一个token，一个英文单词为一个token，空格不计算，标点符号计入计算
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

        
#分词
def GetTokenPos(wd):
    TCSegment(seghandle, wd)
    rescount = TCGetResultCnt(seghandle)
    SegArray = []
    for i in range(rescount):
        wordpos = TCGetAt(seghandle, i)     
        bcw = wordpos.bcw
        pos = wordpos.pos
        word = wordpos.word
        #if pos == 34 or word in stop_words:
        #    continue
        SegArray.append((word, pos))
    return SegArray 
#无意义词或者词组或者口语话词
meanless_words = [ "hi", "hello", "the", "nihao", "你好", "呵呵", "赫赫", "嗬嗬", "哈哈", "谢谢", "感谢", "这", "是", "吗"
                "吧", "了", "呢", "呀", "阿", "哎", "唉", "艾", "恩", "嗯", "唔", "哦", "噢", "喔", "啊", "哈", "嘿",
                "呃", "噢", "把", "嗨", "有", "亲", "哼", "很", "哇", "哇哇", "么", "么么", "哒", "喂", "哦哦",
                "哎呀", "哈喽", "嘿呀", "的", "de", "阿", "噗", "嘿嘿", "嗯嗯",  "不", "晕", "了",
                "啦", "吧", "没", "哇", "嗯哼", "是啊", "个", "他", "啥", "喔", "啦啦", "就", "好呀",
                "嘎嘎", "呼呼", "哦", "额额", "嘛呢", "吖", "哼", "嗯哪", "啊哈", "嗨嗨", "呢", "她", "再", "啦",
                "嗯呢", "蒽蒽", "我靠", "咳咳", "恩呢", "奥", "有啊", "呃呃", "摁", "么", "嗯啊", "哎呀", "是么", "哇塞",
                "噢噢", "咳", "恩啊", "吼吼", "咯", "喂喂", "好了", "耶", "啊哦", "纳尼", "呃", "诶", "阿", "噢",
                "赫赫", "喔喔", "好滴", "我勒", "哦了", "哼哼", "好哦", "咋了", "嘎", "恩哼", "哟", "啥啊", "哇哦", "咦",
                "呃", "哎", "恩", "额", "嗨", "恩", "嗯", "哦", "啊", "嗯嗯", "恩恩", "额",  "嘿嘿", "嘻嘻",
                "帮我", "请你", "拜拜", "是啊", "傻逼", "是的", "好的", "再见", "额", "好啊", "谢谢", "恩",
                
               ]
chengdu_words = ["很", "更", "又", "最", "太", "越", "挺", "想","会", 
                 "可以", "能", "很", "非常","更加", "越来越", "越发", "总之","能够",
                "与", "和", "或", "关于", "至于", "就是", "全部", "所有",
                 "好吗", "还有","没有了", "没了",]
