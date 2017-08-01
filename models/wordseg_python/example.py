#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------#
# example.py
# 
# 腾讯研究院中文处理组
# vim: ts=4 sw=4 sts=4 et tw=78:
# Wed Jan 21 18:08:42 CST 2009
#
#----------------------------------------------------------------------------#

""" python 调用 TCWordSeg 的接口
"""

#----------------------------------------------------------------------------#

import sys
from TCWordSeg import *

dict_dir    = './dict'     # 词典数据目录
input_file  = './test.in'  # 输入待切分文件
output_file = 'test.out'    # 输出切分结果文件

#----------------------------------------------------------------------------#
def do_seg(dict, input, output):
    """ do segmentation
    """
    TCInitSeg(dict)
    SEG_MODE = TC_U2L|TC_POS|TC_S2D|TC_U2L|TC_T2S|TC_ENGU|TC_CN
    seghandle = TCCreateSegHandle(SEG_MODE)

    outs = open(output, 'w')
    for line in open(input):
        line = line.rstrip()
        TCSegment(seghandle, line)
        rescount = TCGetResultCnt(seghandle)
        line_seg = ''
        for i in range(rescount):
            # 只输出词
            #word = TCGetWordAt(seghandle, i)
            #line_seg += word + ' '

            # 输出词和词性
            wordpos = TCGetAt(seghandle, i);
            word = wordpos.word
            pos  = wordpos.pos
            line_seg += '%s/%s ' %(word, pos)

        print >>outs, line_seg

    TCCloseSegHandle(seghandle)
    TCUnInitSeg()

#----------------------------------------------------------------------------#

if __name__ == '__main__':
    do_seg(dict_dir, input_file, output_file)

#----------------------------------------------------------------------------#

