#coding=utf-8
from scipy import stats
from tools import remove_rep
import numpy as np
from util import tokenize_word
from util import TOKEN
import logging
blacklist = [ 
    ["支持一下"],
    ["微博"],
    ["哈哈哈"],
    ["谢谢", "关注"],
    ["感谢", "关注"],
    ["感谢", "支持"],
    ["可怜的孩子"],
    ["原来是这样"]
]
def lp(length, alpha):
    #return pow(5 + length / 6, alpha)

    #lp_ratio = pow((0.1 + length) / 4.1, alpha)
    lp_ratio = pow((1 + length) / 6.0, alpha)
    return lp_ratio 

def cp(attn_scores, beta):
    sum_dec = np.sum(attn_scores, 0) / len(attn_scores)
    return beta * stats.entropy(sum_dec) #(-1) * beta * stats.entropy(sum_dec) 
    #print [math.log(min(e, 1.0)) for e in sum_dec]
    #return beta * sum([math.log(min(e, 1.0)) for e in sum_dec])

def handle_beam_out(out, beam_splits):
    outputs = []
    probs = []
    attns = []

    for n in range(len(out["beam_symbols"])):
        beam_symbols = out["beam_symbols"][n]
        beam_parents = out["beam_parents"][n]

        beam_ends = out["beam_ends"][n]
        beam_end_parents = out["beam_end_parents"][n]
        beam_end_probs = out["beam_end_probs"][n]
        beam_attns = out["beam_attns"][n]

        k = len(beam_symbols[0]) - 1
        example_outputs = [] 
        example_probs = []
        example_attns = []

        while k >= 0:
            for i in range(len(beam_ends)):
                if beam_ends[i][k] != 1 :
                    continue
                idx = beam_end_parents[i][k]
                example_probs.append(beam_end_probs[i][k])
                res = []
                step_attn = []
                j = k - 1
                while j >= 0:
                    res.append(beam_symbols[idx][j])
                    step_attn.append(beam_attns[idx][j])
                    idx = beam_parents[idx][j]
                    j -= 1
                res.reverse()
                step_attn.reverse()
                example_outputs.append(res)
                example_attns.append(step_attn)
            k -= 1 
        outputs.append(example_outputs)
        probs.append(example_probs)
        attns.append(example_attns)
    return outputs, probs, attns


def rank(res_list):
    # model_name, final, info, gnmt_score, probs[n], lp_ratio, cp_score, enc_attn_scores   
    sorted_res = sorted(res_list, key=lambda x:x[3], reverse=True)
    gap = 4.0
    if len(res_list) < 2:
        return res_list
    if res_list[0][3] - res_list[1][3] > gap and res_list[1][3] > -999:
        tmp = list(res_list[0])
        tmp[3] = -1000
        tmp[2] = "the first ans exceeds the second more than 4"
        res_list[0] = tmp
    sorted_res = sorted(res_list, key=lambda x:x[3], reverse=True)
    return sorted_res

def score_couplet(model_name, poem_sen, match_sens, probs):
    poem_ws, poem_pos = tokenize_word(poem_sen, need_pos=True)
    poem_sen_uni = poem_sen.decode("utf-8")
    poem_pos_str = " ".join([str(e) for e in poem_pos])

    # for mophological similarity
    match_sen_cans = []
    match_ws_cans = []
    match_pos_cans = []
    prob_cans = []
    for i, each in enumerate(match_sens):
        match_sen_uni = each.decode("utf-8")[0:len(poem_sen_uni)]
        if len(match_sen_uni) != len(poem_sen_uni):
            continue

        moph_match = True 
        for j in range(1, len(match_sen_uni)): 
            if match_sen_uni[j] == match_sen_uni[j - 1] and poem_sen_uni[j] != poem_sen_uni[j - 1]:
                moph_match = False 
                logging.info("bad moph %s" % each)
                break
            if match_sen_uni[j] != match_sen_uni[j - 1] and poem_sen_uni[j] == poem_sen_uni[j - 1]:
                logging.info("bad moph %s" % each)
                moph_match = False 
                break
        for j in range(len(match_sen_uni)):
            if match_sen_uni[j] == poem_sen_uni[j]:
                logging.info("bad moph %s" % each)
                moph_match = False 
                break

        if moph_match: 
            w, pos = tokenize_word(each, need_pos=True)
            match_ws_cans.append(w)
            match_pos_cans.append(pos)
            match_sen_cans.append(each)
            prob_cans.append(probs[i])

    # for pos similarity (NOT PRECISE)
    for i, ws in enumerate(match_ws_cans):
        if not ws: 
            continue
        if len(ws) != len(poem_ws):
            continue
        match_pos_str = " ".join([str(e) for e in match_pos_cans])
        if match_pos_str == poem_pos_str: 
            prob_cans[i] += 100

    # for tone match
    return match_sen_cans, prob_cans 

def score_with_prob_attn(model_name, ans, probs, attns, alpha=0.9, beta=0.1, is_ch=True, average_across_len=False):
    res = []
    for n in range(len(ans)):
        words = []
        dup = {}
        dupcount = 0
        isdup = False
        inbl = False
        count = 0
        punc_cnt = 0
        ch_bytes_cnt = 0
        for j, w in enumerate(list(ans[n])):
            if w == "_EOS":
                break
            if w == "":
                continue

            # for different language, handle dups and count puncs 
            if is_ch:
                for c in w.decode("utf-8"):
                    if c in [u"。", u"，", u"？", u"！", u",", u"!", u"?", u"."]:
                        punc_cnt += 1
                    else:
                        ch_bytes_cnt += 1 
                    if c not in dup:
                        dup[c] = 0
                    dup[c] += 1
            else:
                if w.decode("utf-8") in [u",", u"!", u"?", u"."]:
                    punc_cnt += 1
                if w not in dup:    
                    dup[w] = 0
                dup[w] += 1

            if j == 0 or w != words[-1]:
                count == 0
                words.append(w)
            elif w.decode("utf-8") in [u"。", u"，", u"？", u"！", u",", u"!", u"?", u"."] and w == words[-1] and count < 2:
                words.append(w)
                count += 1
            elif w.decode("utf-8") not in [u"。", u"，", u"？", u"！", u"!", u",", u"?", u"."] and len(w.decode("utf-8")) > 1 and w == words[-1]:
                isdup = True
                break
        #words = remove_rep(words)
        for each in dup: 
            if each.decode("utf-8") not in [u"。", u"，", u"？", u"哈", u".", u",", u"!", u"?", u"'"] and dup[each] > 2:
                isdup = True
                break
            if dup[each] > 1:
                dupcount += 1
                #if (dupcount >= len(dup) / 2) and dupcount >= 2:
                if is_ch == True and dupcount >= 2:
                    isdup = True
                    break
        #attn_score = np.sum(attns[n], 0) / len(attns[n])
        enc_attn_scores = " "
        #enc_attn_scores = " ".join(["%.3f" % round(a, 4) for a in attn_score])
        #enc_attn_scores = (list(attn_score[0:36]), list(attn_score[36:]))
        #enc_attn_scores = "none"

        #lp_ratio = lp(ch_bytes_cnt, alpha)
        lp_ratio = lp(len(words), alpha)
        #lp_ratio = len(words) * 1.0  
        cp_score = 0.0 #cp(attns[n], beta)
        info = ""
        final = "".join(words) if is_ch else " ".join(words)
        seged = " ".join(words)
        
        # check NE
        NE = "" 
        if is_ch:
            _, pos = tokenize_word(final, True)
            for p in pos:
                if TOKEN[p] == "TC_NR" or TOKEN[p] == "TC_NRF" or TOKEN[p] == "TC_NRG":
                    NE = TOKEN[p] 
                    break
            
        # check blacklist
        inbl = False
        for each in blacklist:
            hit = True 
            for p in each:
                if final.find(p) == -1: 
                    hit = False
                    break
            if hit:
                inbl = True
                break

        if "_PAD" in words:
            gnmt_score = -1000
            info = "_PAD in sentence"
        elif "_UNK" in ans[n]:
            gnmt_score = -1000
            info = "_UNK found"
        elif isdup:
            gnmt_score = -1000
            info = "dup words exceeds half of total"
        elif len(final) < 8:
            gnmt_score = -1000
            info = "too short"
        elif punc_cnt > len(words) / 2.0:
            gnmt_score = -1000
            info = "too many puncs"
        elif inbl:
            gnmt_score = -1000
            info = "in blacklist"
        elif NE != "":
            gnmt_score = -1000
            info = "NE %s detected !" % NE
        else: 
            gnmt_score = probs[n] / float(len(words)) if average_across_len else probs[n]
            #gnmt_score = probs[n] / lp_ratio + cp_score
        res.append((model_name, final, info, gnmt_score, probs[n], lp_ratio, cp_score, enc_attn_scores, seged))
    return res
