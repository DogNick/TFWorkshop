#coding=utf-8

'''
	author: cuiyanyan
	功能: 腾讯分词
'''

import os
import sys
import pdb
import time
import logging
import traceback

curr_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(curr_dir, '../../lib'))
#sys.path.append(os.path.join(curr_dir, '../'))

from tencent_wordseg.TCWordSeg import *


time_format = '%Y-%m-%d %H:%M:%S'
dict_dir = os.path.join(curr_dir, './tencent')


def init_tencent_seg(dict_dir):
	'''
		功能: 初始化分词，分词结果都返回空
		params:
			dict_dir: 腾讯分词加载的数据路径
		return:
			seg_flag: 分词初始化是否成功
			seghandle: 分词句柄
	'''
	seg_flag = True
	seghandle = None
	try:
		if TCInitSeg(dict_dir):
			SEG_MODE = TC_S2D|TC_CN|TC_POS|TC_T2S|TC_U2L|TC_CLS|TC_USR|TC_LN
			seghandle = TCCreateSegHandle(SEG_MODE)
		else:
			seg_flag = False
	except Exception, e:
		seg_flag = False
		logging.warning('[common_ERROR] [tencent_segment] [init_tencent_seg] [%s]' % (time.strftime(time_format)))
		logging.warning(traceback.format_exc(e))
	return seg_flag, seghandle


#init
begin = time.time()
seg_flag, seghandle = init_tencent_seg(dict_dir)
cost = str(int(round(time.time()-begin, 3)*1000))
logging.info('[common_init] [tencent_segment] [%s] [cost=%sms]' % (time.strftime(time_format), cost))


def seg_with_pos(query_uni):
	'''
		功能: 分词
		params:
			query_uni: 需要分词的句子，编码Unicode
		return:
			line_seg: 返回分词和词性，编码Unicode
	'''
	line_seg = []
	if seg_flag == False:
		return line_seg
	if query_uni == '' or query_uni == None:
		return line_seg
	try:
		query_gbk = query_uni.encode('gbk', 'ignore')
		TCSegment(seghandle, query_gbk)
		rescount = TCGetResultCnt(seghandle)
		for i in range(rescount):
			wordpos = TCGetAt(seghandle, i);
			word = wordpos.word.decode('gbk')
			pos = wordpos.pos
			line_seg.append((word, pos))
	except Exception, e:
		logging.warning('[common_ERROR] [tencent_segment] [seg_with_pos] [%s]' % (time.strftime(time_format)))
		logging.warning(traceback.format_exc(e))
	return line_seg


def seg_return_string(query_uni):
	'''
		功能: 分词
		params:
			query_uni: 需要分词的句子，编码Unicode
		return:
			string_seg: 返回分词，以空格隔开，编码gbk
	'''
	string_seg = ''
	if seg_flag == False:
		return string_seg
	if query_uni == '' or query_uni == None:
		return string_seg
	try:
		query_gbk = query_uni.encode('gbk', 'ignore')
		TCSegment(seghandle, query_gbk)
		rescount = TCGetResultCnt(seghandle)
		for i in range(rescount):
			wordpos = TCGetAt(seghandle, i);
			word = wordpos.word
			pos = wordpos.pos
                        string_seg += word + ' '
			#if pos != 34: # rm punctation
			#	string_seg += word + ' '
		string_seg = string_seg[:-1]
	except Exception, e:
		logging.warning('[common_ERROR] [tencent_segment] [seg_return_string] [%s]' % (time.strftime(time_format)))
		logging.warning(traceback.format_exc(e))
	return string_seg


def uninit_seg(seghandle):
	if seg_flag:
		TCCloseSegHandle(seghandle)
		TCUnInitSeg()


'''
******************************************分词切分开关**********************************************
TC_ENGU          0x00000001         数字英文串连续时使用小粒度模式（优先级高于TC_GU)
TC_GU            0x00000002         整个分词系统使用小粒度模式
TC_POS           0x00000004         进行词性标注
TC_USR           0x00000008         使用用户自定义词
TC_S2D           0x00000010         进行全角半角转换
TC_U2L           0x00000020         进行英文大小写转换
TC_CLS           0x00000040         标注用户自定义词分类号
TC_RUL           0x00000080         使用规则
TC_CN            0x00000100         使用人名识别
TC_T2S           0x00000200         使用繁体转简体
TC_PGU           0x00000400         人名小粒度开关
TC_LGU           0x00000800         地名小粒读开关
TC_SGU           0x00001000         带后缀字的词小粒度开关
TC_CUT           0x00002000         短语模式切分开关
TC_TEXT          0x00004000         篇章信息分词开关
TC_CONV          0x00008000         字符！，： ；？全半角转换关闭开关
TC_WMUL          0x00010000         共享单字片段采用Multi-Seg
TC_PMUL          0x00020000         真组合歧义片段采用Multi-Seg
TC_ASC           0x00040000         ASCII字符串作为整体切分
TC_SECPOS        0x00080000         使用二级词性
TC_GBK           0x00100000         字符串编码格式:GBK编码
TC_UTF8          0x00200000         字符串编码格式:UTF-8编码
TC_NEW_RES       0x00400000         用于生成新的接口形式即可同时返回,细切分、粗切分、用户自定义词、短语词、同义词
TC_SYN           0x00800000         用于生成同义词
TC_LN            0x01000000         地名识别开关
TC_WGU           0x02000000         共享单字片段采用细切分
****************************************************************************************************

******************************************分词词性**************************************************
TC_UNK           0                  未知词性
TC_A             1                  形容词
TC_AD            2                  副形词
TC_AN            3                  名形词
TC_B             4                  区别词
TC_C             5                  连词
TC_D             6                  副词
TC_E             7                  叹词
TC_F             8                  方位词
TC_G             9                  语素词
TC_H             10                 前接成分
TC_I             11                 成语
TC_J             12                 简称略语
TC_K             13                 后接成分
TC_L             14                 习用语
TC_M             15                 数词
TC_N             16                 名词
TC_NR            17                 人名
TC_NRF           18                 姓
TC_NRG           19                 名
TC_NS            20                 地名
TC_NT            21                 机构团体
TC_NZ            22                 其他专名
TC_NX            23                 非汉字串
TC_O             24                 拟声词
TC_P             25                 介词
TC_Q             26                 量词
TC_R             27                 代词
TC_S             28                 处所词
TC_T             29                 时间词
TC_U             30                 助词
TC_V             31                 动词
TC_VD            32                 副动词
TC_VN            33                 名动词
TC_W             34                 标点符号
TC_X             35                 非语素字
TC_Y             36                 语气词
TC_Z             37                 状态词
TC_AG            38                 形语素
TC_BG            39                 区别语素
TC_DG            40                 副语素
TC_MG            41                 数词性语素
TC_NG            42                 名语素
TC_QG            43                 量语素
TC_RG            44                 代语素
TC_TG            45                 时语素
TC_VG            46                 动语素
TC_YG            47                 语气词语素
TC_ZG            48                 状态词语素
TC_SOS           49                 开始词
TC_WWW           50                 URL
TC_TELE          51                 电话号码
TC_EMAIL         52                 email
TC_EOS           55                 结束词
****************************************************************************************************
'''


if __name__ == '__main__':
	'''tc debug'''
	#print dict_dir 
	querys = [u'汪仔，你多大了, 有女朋友吗']
	for query in querys:
		segs = seg_with_pos(query)
		seg = seg_return_string(query)
		print '>>>>>>>>'
		print query.encode('utf-8')
		for i in range(len(segs)):
			print segs[i][0].encode('utf-8') + '\t' + str(segs[i][1])
		print seg.decode('gbk').encode('utf-8')
		print '\n\n'
