#coding=utf-8
import pysrt
import codecs
import gevent
import threading
import sys
import os
import re


def clean(text):
    text = re.sub("\(.*\)", "", text)
    text = re.sub("\[.*?\]", "", text)
    if re.search("[^a-zA-Z',!?: .\t\n]", text):
        return ""
    if len(text.split()) < 2:
        return ""
    text = re.sub("\.\.+| +|\t+", " ", text)
    text = re.sub("^[^:]: ", "", text)
    text = re.sub("\.", " . ", text)
    text = re.sub("\?", " ? ", text)
    text = re.sub("!", " ! ", text)
    text = re.sub(":", " : ", text)
    text = re.sub(",", " , ", text)

    #text = re.sub("<i>.*?</i>", "", text)
    #text = re.sub("<font [^>]*>.*?</font>", "", text)
    #if re.search(u"[\u4e00-\u9fa5]| \- |^\-", text):
    #    text = ""
    text = re.sub("\.\.\.|\n", " ", text)
    return text.lower()


def proc(pid, selected, fout, lock, counts):
    pt_call = " *(me|I|you|she|her|us|i|You|Me|us|them|they)([^a-zA-Z]|$)"#[ !\?:,:\n]*"
    for each in selected:
        if not os.path.isfile(each):
            continue
        subs = pysrt.open(each, encoding="iso-8859-1")
        # concat
        i = 0
        sens = []
        while (i < len(subs)):
            #print subs[i].text 
            #raw_input()
            #if subs[i].text.find("\t") != -1:
                #print subs[i].text, subs[i].text.find("\t")
                #raw_input()
            j = i
            while j < len(subs) -1 and len(subs[j].text) >= 3 and subs[j].text[-3:] == "...":
                j = j + 1
            text = "".join([each.text for each in subs[i:j+1]])
    
            text = clean(text) 
            #print "[done]########", text
            sens.append((subs[i].start, subs[j].end, text))
            i = j + 1
    
        ## jump
        cut = 0
        while(cut < len(sens)):
            p = sens[cut][2]
            if re.search(pt_call, p):
                #print "[HIT]", p
                break 
            cut = cut + 1
        sens = sens[cut:]
    
        # select
        for i in range(len(sens) - 1):
            if sens[i][2] == "" or sens[i + 1][2] == "":
                continue
            t1 = sens[i][1].hours * 3600000 + sens[i][1].minutes * 60000 + sens[i][1].seconds
            t2 = sens[i+1][0].hours * 3600000 + sens[i+1][0].minutes * 60000 + sens[i+1][0].seconds
            if t1 + 700 > t2:
                print "%s ---> %s" % (sens[i][2], sens[i + 1][2])
                raw_input()
                lock.acquire() 
                counts[pid] += 1
                if counts[pid] % 10000 == 0:
                    print_str = "\n\r".join(["Pid %d proc: %d" % (i, cnt) for i, cnt in enumerate(counts)])
                    print "%s\r" % print_str,
                #fout.write("%s\t%s\n" % (sens[i][2].encode("utf-8"), sens[i+1][2].encode("utf-8")))
                lock.release() 

    print_str = "\n\r".join(["Pid %d proc: %d" % (i, cnt) for i, cnt in enumerate(counts)])
    print "%s\r" % print_str,

if __name__ == "__main__":
    proc_num = int(sys.argv[1])
    pt_en = "\.en\.srt|\.En\.srt|\.eng\.srt|\.Eng\.srt|\.English\.srt|\.英文\.srt"
    name = sys.argv[1] 
    outname = "srt.pair" 
    a = os.listdir(name)
    selected = []
    for each in a:
        dist_dir = os.path.join(name, each)
        b = os.listdir(dist_dir)
        for f in b: 
            if re.search(pt_en, f):
                selected.append(os.path.join(dist_dir, f))
        #print each
        #print "\n".join(selected)
        #print
        #raw_input()
    print len(selected)
    lock = threading.Lock()
    fout = codecs.open(outname, "w")
    counts = [0] * proc_num 
    offset = len(selected) / proc_num
    threads = []
    pid = 0
    proc(pid, selected, fout, lock, counts)
    #for start in range(0, len(selected), offset):
    #    threads.append(gevent.spawn(proc, pid, selected[start:start+offset], fout, lock, counts))
    #    #threads.append(threading.Thread(target=proc, args=(pid, selected[start:start+offset], fout, lock, counts)))
    #    pid += 1
    ##for th in threads:
    ##    th.start()
    ##for th in threads:
    ##    threading.Thread.join(th)
    #gevent.joinall(threads) 
