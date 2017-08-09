#!/usr/bin/python
#coding=utf8
import re
import requests
import sys
import time
import urllib
from conf import *

sub_catagory_count = {}
query_frm_count = {}
query_frm_user = {}
record_lst = []
by_hour = {}
for hour in range(0, 24):
	by_hour[hour] = {}
	by_hour[hour]['group'] = set()
	by_hour[hour]['user'] = set()
	by_hour[hour]['message'] = 0

def read(filein):
	for line in open(filein):
		if line.find('[#USR#]') == 0: 
			record = re.findall(r'\[#USR#\]\[UUID:(.+?)\]\[timestamp:(.+?)\]\[groupid=(.+?)\]\[uid=(.+?)\]\[query=(.*)\]\[data=(.+?)\]', line)[0]
			record = ['USR'] + list(record)
			record_lst.append(record)
			# update statistics
			time_str = time.strftime('%H:%M:%S', time.localtime(float(record[2]))) # record[2] is timestamp
			hour = int(time_str.split(':')[0])
			by_hour[hour]['group'].add(record[3]) # record[3] is groupid
			by_hour[hour]['user'].add(record[4]) # record[4] is uid
			by_hour[hour]['message'] += 1

		if line.find('[#BOT#]') == 0:
			record = re.findall(r'\[#BOT#\]\[UUID:(.+?)]\[timestamp:(.+?)]\[groupid=(.+?)]\[uid=(.+?)\]\[answer=(.+?)\]\[from=(.+?)\]', line)[0]
			record = ['BOT'] + list(record)
			record_lst.append(record)

		if line.find('[#ANALYSE-VERBOSE#]') == 0:
			items = re.findall('\[#ANALYSE-VERBOSE#\]\[UUID:(.+?)\]\[timestamp:(.+?)\]\[time:(.+?)\]\[groupid=(.+?)\]\[uid=(.+?)\]\[query=(.*)\]\[data=(.+?)\]\[answer=(.+?)\]\[from=(.+?)\]', line)[0]
			uid = items[4]
			query = items[5].replace('\t', '    ')
			frm = items[8]
			key = query + '\t' + frm
			if not query_frm_count.has_key(key):
				query_frm_count[key] = 0
			query_frm_count[key] += 1
			if not query_frm_user.has_key(key):
				query_frm_user[key] = set()
			query_frm_user[key].add(uid)
	return

def dispatch(outfile_prefix):
	out_op = open(outfile_prefix + '-freq-operation', 'w')
	out_em = open(outfile_prefix + '-freq-emoji', 'w')
	out_vt = open(outfile_prefix + '-freq-vertical', 'w')
	out_df = open(outfile_prefix + '-freq-defaults', 'w')
	out_un = open(outfile_prefix + '-freq-unhandled', 'w')
	out_ed = open(outfile_prefix + '-freq-QA-or-CHAT', 'w')


	sub_catagory_count['total'] = 0

	'''
	# 以query绝对次数为标准
	for key in query_frm_count.keys():
		if query_frm_count[key] <= 2:
			del query_frm_count[key]
	lst = sorted(query_frm_count.items(), key=lambda x:x[1], reverse=True)
	'''

	# 以query用户数目为标准
	lst = []
	for key in query_frm_user.keys():
		if len(query_frm_user[key]) <= 1:
			del query_frm_user[key]

	query_frm_lenuser = {k : len(v) for k, v in query_frm_user.items()}
	lst = sorted(query_frm_lenuser.items(), key=lambda x:x[1], reverse=True)

	for record in lst:
		query, frm = record[0].split('\t')
		count = record[1]
		sub_catagory_count['total'] += count
		if not sub_catagory_count.has_key(frm):
			sub_catagory_count[frm] = 0
		sub_catagory_count[frm] += count
		
		line = '%s\t%s\t%s' % (frm, query, count)
		if frm == 'precision_chat' or frm == 'whitelist_pattern' or frm == 'blacklist':
			out_op.write(line + '\n')
		elif frm == 'emoji_reply':
			out_em.write(line + '\n')
		elif frm == 'yyzs' or frm == 'universal_time' or frm == 'weather' or frm == 'traslation':
			out_vt.write(line + '\n')
		elif frm.find('default_') == 0:
			out_df.write(line + '\n')
		elif query.find('[') != -1 or query.find(']') != -1:
			out_un.write(line + '\n')
		else:
			out_ed.write(line + '\n')
		
	out_op.close()	
	out_em.close()	
	out_vt.close()	
	out_df.close()	
	out_un.close()	
	out_ed.close()	
	return

def get_ratio_table():
	table = '''
	<h3>占比统计</h3>
	<table id="rounded-corner">
		<thead>
			<tr>
				<th class="green" scope="col">总类别</th>
				<th class="green" scope="col">占比</th>
				<th class="green" scope="col">细分类别</th>
				<th class="green" scope="col">占比</th>
			</tr>
		</thead>
		<tbody>
	'''
	for catagory in catagory_lst:
		
		count = 0
		lst = []
		for sub_catagory in catagory_map[catagory]:
			try:
				count += sub_catagory_count[sub_catagory]
				lst.append((sub_catagory, sub_catagory_count[sub_catagory]))
			except:
				count += 0
		if len(lst) > 0:
			lst2 = sorted(lst, lambda x, y: cmp(x[1], y[1]), reverse=True)
			span = len(lst2)
			table += '<tr><td rowspan="%d">%s</td><td rowspan="%d">%.2f%%</td><td>%s</td><td>%.2f%%</td>' % (span, cn_name_map[catagory], span, float(count) / sub_catagory_count['total'] * 100, cn_name_map[lst2[0][0]], float(lst2[0][1]) / sub_catagory_count['total'] * 100)
			for i in range (1, len(lst2)):
				table += '<tr><td>%s</td><td>%.2f%%</td>' % (cn_name_map[lst2[i][0]], float(lst2[i][1]) / sub_catagory_count['total'] * 100)
	table += '</tbody></table>'
	return table

def get_traffic_table():
	total_group = set()
	total_user = set()
	total_message = 0
	table = '''
	<h3>流量统计</h3>
	<table id="rounded-corner">
	'''

	# AM
	table += '<thead><tr><th class="blue" scope="col">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>'
	for hour in range(0, 12):
		table += '<th class="red" scope="col">%02d</th>' % hour
	table += '</tr></thead>'

	table += '<tr><td>活跃群</td>'
	for hour in range(0, 12):
		table += '<td>% 4d</td>' % len(by_hour[hour]['group'])
	table += '</tr>'

	table += '<tr><td>活跃用户</td>'
	for hour in range(0, 12):
		table += '<td>% 4d</td>' % len(by_hour[hour]['user'])
	table += '</tr>'

	table += '<tr><td>消息量</td>'
	for hour in range(0, 12):
		table += '<td>% 4d</td>' % by_hour[hour]['message']
	
	# PM
	table += '<thead><tr><th class="blue" scope="col">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>'
	for hour in range(12, 24):
		table += '<th class="red" scope="col">%02d</th>' % hour
	table += '</tr></thead>'

	table += '<tr><td>活跃群</td>'
	for hour in range(12, 24):
		table += '<td>% 4d</td>' % len(by_hour[hour]['group'])
	table += '</tr>'

	table += '<tr><td>活跃用户</td>'
	for hour in range(12, 24):
		table += '<td>% 4d</td>' % len(by_hour[hour]['user'])
	table += '</tr>'

	table += '<tr><td>消息量</td>'
	for hour in range(12, 24):
		table += '<td>% 4d</td>' % by_hour[hour]['message']
	table += '</tr>'

	# DAY
	for hour in range(0, 24):
		total_group |= by_hour[hour]['group']
		total_user |= by_hour[hour]['user'] # note: users from different group must not share uid
		total_message += by_hour[hour]['message']

	table += '</table>'
	table += '<h4>&nbsp;活跃群:&nbsp;%d&nbsp;&nbsp;活跃用户:&nbsp;%d&nbsp;&nbsp;消息总量:&nbsp;%d</h4>' % (len(total_group), len(total_user), total_message)



	return table

def send_email(names_to_email, date):

	html = '<html><head>%s</head><body>%s<br>%s</body></html>' % (css, get_ratio_table(), get_traffic_table())

	uid = 'ps_id@sogou-inc.com'
	fr_name = '搜狗汪仔'.decode('utf8', 'ignore').encode('gbk', 'ignore')
	fr_addr = 'ps_id@sogou-inc.com'
	title = ('汪仔QQ热聊日报 - %s' % date).decode('utf8', 'ignore').encode('gbk', 'ignore')

	html = html.decode('utf8', 'ignore').encode('gbk', 'ignore')
	send_to = ';'.join([k + '@sogou-inc.com' for k in names_to_email.split(',')])
	print send_to

	url = 'http://mail.portal.sogou/portal/tools/send_mail.php?uid=%s&fr_name=%s&fr_addr=%s&title=%s&body=%s&mode=html&maillist=%s&attname=&attbody=' % (uid, urllib.quote(fr_name), fr_addr, urllib.quote(title), urllib.quote(html), send_to)
	requests.get(url)


def write_neat(filename):
	sorted_record_lst = sorted(record_lst, key = lambda x:(x[3], float(x[2])))
	out = open(filename, 'w')
	last_groupid = 'x'
	group_cnt = 1
	user_map = {}
	for record in sorted_record_lst:
		role, uuid, timestamp, groupid, uid, content, debug = record
		date_str = time.strftime('%Y-%m-%d', time.localtime(float(timestamp)))
		time_str = time.strftime('%H:%M:%S', time.localtime(float(timestamp)))
		if groupid != last_groupid:
			out.write('\n\n\n\n==== Group #%04d DATE:%s ID:%s ====\n\n' % (group_cnt, date_str, groupid))
			last_groupid = groupid
			group_cnt += 1
			user_map = {}
		
		if uid not in user_map:
			user_map[uid] = len(user_map.keys()) + 1

		if role == 'USR':
			out.write('%s\tUSR%02d:\t%s\n' % (time_str, user_map[uid], content))
		if role == 'BOT':
			out.write('%s\tBOT:\t@USR%02d %s |%s\n' % (time_str, user_map[uid], content, debug))

	out.close()
	return
		
		

if __name__ == '__main__':

	if len(sys.argv) != 3 and len(sys.argv) != 4 and len(sys.argv) != 6:
		print 'usage: %s [input] [output_prefix] [optional: 1 for write neat file)] [optional: comma seperated names to email] [optional: email date tag]' % sys.argv[0]
		sys.exit()

	read(sys.argv[1])
	dispatch(sys.argv[2])
	if len(sys.argv) > 3 and sys.argv[3] == '1':
		write_neat(sys.argv[2] + '-neat.txt')
	if len(sys.argv) > 5:
		send_email(sys.argv[4], sys.argv[5])
