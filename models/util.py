#coding=utf-8
import logging as log
import codecs
import os
import re
import sys
import numpy as np
from wordseg_python import Global


TOKEN = {
	1:"TC_A", # 形容词
	2:"TC_AD", # 副形词
	3:"TC_AN", # 名形词
	4:"TC_B", # 区别词
	5:"TC_C", # 连词
	6:"TC_D", # 副词
	7:"TC_E", # 叹词
	8:"TC_F", # 方位词
	9:"TC_G", # 语素词
	10:"TC_H", # 前接成分
	11:"TC_I", # 成语
	12:"TC_J", # 简称略语
	13:"TC_K", # 后接成分
	14:"TC_L", # 习用语
	15:"TC_M", # 数词

	16:"TC_N", # 名词
	17:"TC_NR", # 人名
	18:"TC_NRF", # 姓
	19:"TC_NRG", # 名
	20:"TC_NS", # 地名
	21:"TC_NT", # 机构团体
	22:"TC_NZ", # 其他专名
	23:"TC_NX", # 非汉字串
	24:"TC_O", # 拟声词
	25:"TC_P", # 介词
	26:"TC_Q", # 量词
	27:"TC_R", # 代词
	28:"TC_S", # 处所词
	29:"TC_T", # 时间词
	30:"TC_U", # 助词
	31:"TC_V", # 动词
	32:"TC_VD", # 副动词
	33:"TC_VN", # 名动词
	34:"TC_W", # 标点符号
	35:"TC_X", # 非语素字
	36:"TC_Y", # 语气词
	37:"TC_Z", # 状态词
	38:"TC_AG", # 形语素
	39:"TC_BG", # 区别语素
	40:"TC_DG", # 副语素
	41:"TC_MG", # 数词性语素
	42:"TC_NG", # 名语素
	43:"TC_QG", # 量语素
	44:"TC_RG", # 代语素
	45:"TC_TG", # 时语素
	46:"TC_VG", # 动语素
	47:"TC_YG", # 语气词语素
	48:"TC_ZG", # 状态词语素
	49:"TC_SOS", # 开始词
	0:"TC_UNK", # 未知词性
	50:"TC_WWW", # URL
	51:"TC_TELE", # 电话号码
	52:"TC_EMAIL" # email
}

RepSet = {
	# 0:"TC_UNK", # 未知词性
	#15:"TC_M", # 数词
	50:"TC_WWW", # URL
	51:"TC_TELE", # 电话号码
	52:"TC_EMAIL", # email
	#23:"TC_NX" # 非汉字串
}
DISCARD = {
    #u"我也是":0,
    #u"我也不知道":0,
    #u"我也想知道":0
}

graphlg = log.getLogger("graph")


def tokenize_char(sentence):  
    sentence = re.sub("\t*\s*", "", sentence)
    tokens = []
    for each in re.finditer(u"[0-9.０１２３４５６７８９a-zA-Z:?!\"-]+|[\u4e00-\u9fa5，。：？！]", sentence.decode("utf-8", "ignore")): 
        tokens.append(each.group().encode("utf-8"))

    #sentence = re.sub(u"[0-9.０１２３４５６７８９]+", "NUM", sentence)
    if len(tokens) == 0:
        return ""
    return tokens 


def tokenize_word(sentence, need_pos=False):
    tuples = [(w,p) for w,p in Global.GetTokenPos(sentence.decode('utf-8',"ignore").encode('gbk', 'ignore'))]
    res = []
    pos = []
    for t in tuples:
        if t[1] in RepSet:
            res.append(RepSet[t[1]])
        else:
            res.append(t[0].decode('gbk', 'ignore').encode('utf8', 'ignore'))
        if need_pos:
            pos.append(t[1])
    return res, pos 



def filter_no_use(out, DISCARD):
    if out.decode("utf-8") in DISCARD:
        deploylg.info("Deprecated: %s in DISCARD:" % out)
        return ""
    return out 

def filter_dup(post, out):
    if out == "":
        return ""
    words_in = tokenize_word(post, {})
    words_out = tokenize_word(out, {})

    dic_in = {} 
    in_gt_N = {}
    N = 2 
    for each in words_in:
        if each not in dic_in:
            dic_in[each] = 0
        dic_in[each] = dic_in[each] + 1
        if dic_in[each] > N: 
            in_gt_N[each] = dic_in[each] 

    dic_out = {} 
    out_gt_N = {} 
    N = 2 
    for each in words_out: 
        if each not in dic_out:
            dic_out[each] = 0
        dic_out[each] = dic_out[each] + 1
        if dic_out[each] > N: 
            out_gt_N[each] = dic_out[each] 
    
    if len(out_gt_N) > 0:
        for each in out_gt_N:
            if each in in_gt_N:
                continue
            if re.match(u"哈+|啊+|！|，|？|。", each) != None:
                continue
            deploylg.info ("Deprecated: %s more than %d, %s" % (each, N, out))
            out = "" 
    count = 0
    for each in dic_out:
        if dic_out[each] >= 2:
            if len(each) >= 2:
                deploylg.info("Deprecatd: %d chars %s multiple time" % (len(each), each)) 
                out = ""
            count = count + 1
    if count >= len(dic_out) / 2:
        deploylg.info("Deprecated: dups more than half of dic, %s" % out)
        out = ""
    return re.sub(" +", "", out)


_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_ALPHA = b"_ALPHA"
_NUM = b"_NUM"

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
NUM_ID = 4
ALPHA_ID = 5 

graphlg = log.getLogger("graph")
if __name__ == "__main__":
    input_vocab, output_vocab = create_input_output_vocab(sys.argv[1])
    print input_vocab,output_vocab
