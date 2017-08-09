#coding=utf8
css = '''
<style type="text/css">
body
{
	line-height: 1.6em;
}

#rounded-corner
{
	font-family: "Lucida Sans Unicode", "Lucida Grande", Sans-Serif;
	font-size: 18px;
	margin: 5px;
	width: 480px;
	text-align: center;
	border-collapse: collapse;
}
#rounded-corner th.red
{
	padding: 3px;
	font-weight: normal;
	font-size: 18px;
	color: #2B2B2B;
	background: #EA6153;
}
#rounded-corner th.green
{
	padding: 3px;
	font-weight: normal;
	font-size: 18px;
	color: #2B2B2B;
	background: #27AE60;
}
#rounded-corner th.blue
{
	padding: 3px;
	font-weight: normal;
	font-size: 18px;
	color: #2B2B2B;
	background: #2980B9;
}
#rounded-corner td
{
	padding: 8px;
	background: #E9E9E9;
	border-top: 1px solid #fff;
	color: #2B2B2B;
}

</style>
'''

# 统计大类
catagory_lst = ['yunying', 'chat', 'QA', 'vertical', 'other_interaction']
catagory_map = {}
catagory_map['total'] = ['total']
catagory_map['yunying'] = ['precision_chat', 'whitelist_pattern', 'blacklist', 'emoji_reply']
catagory_map['chat'] = ['generate']
catagory_map['QA'] = ['web_search', 'yaoting']
catagory_map['vertical'] = ['yyzs', 'universal_time', 'weather', 'tranlation', 'poem']
catagory_map['other_interaction'] = ['default', 'default_for_atwangzai_query', 'default_for_empty_query', 'default_for_voice_query', 'default_for_sharemusic_query', 'default_for_sharestock_query']

# 大小类的中文名
cn_name_map = {}
cn_name_map["total"]="汇总"
cn_name_map["precision_chat"]="精准"
cn_name_map["generate"]="生成"
cn_name_map["web_search"]="网页"
cn_name_map["emoji_reply"]="emoji"
cn_name_map["whitelist_pattern"]="pattern"
cn_name_map["default_for_atwangzai_query"]="@汪仔"
cn_name_map["yyzs"]="语音助手"
cn_name_map["yaoting"]="姚婷"
cn_name_map["poem"]="诗词"
cn_name_map["default_for_empty_query"]="空query"
cn_name_map["translation"]="翻译"
cn_name_map["blacklist"]="黑名单"
cn_name_map["default_for_voice_query"]="语音"
cn_name_map["default"]="缺省"
cn_name_map["default_for_sharemusic_query"]="分享音乐"
cn_name_map["default_for_sharestock_query"]="分享股票"
cn_name_map["universal_time"]="时间"
cn_name_map["yunying"]="运营"
cn_name_map["chat"]="闲聊"
cn_name_map["QA"]="问答"
cn_name_map["vertical"]="垂类"
cn_name_map["other_interaction"]="其它交互"
cn_name_map["weather"]="天气"
