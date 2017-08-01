#coding=utf-8
import sys
sys.path.insert(0, "models")
sys.path.insert(0, "/search/odin/Nick/_python_build2")
from models import create 
from models import Nick_plan 
import numpy as np
import tensorflow as tf
import codecs
import copy
import logging as log
import math
from scipy import stats
import re
import os

fh = log.StreamHandler()
fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
log.getLogger("graph").addHandler(fh)
log.getLogger("graph").setLevel(log.DEBUG) 
# create training logger
fh = log.StreamHandler()
fh.setFormatter(log.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S"))
log.getLogger("train").addHandler(fh)
log.getLogger("train").setLevel(log.DEBUG)
graphlg = log.getLogger("graph")
trainlg = log.getLogger("train")
train_root = "/search/odin/Nick/GenerateWorkshop/runtime"
def main():
    name = "test/baseline"
    name = "test/p-r_rec-87.labeled"
    name = "test/p-100.test"
    name = "test/stc2.train.good.posts.selected"
    name = "test/400query"
    name = "final"
    #name = "test/2k.post_comment"
    #name = "test/stc2.train.post_comment"
    #name = "test/all_post_uniq"
    #name = "test/_UNK_query"
    name = "test/real_gen_200"
    #name = "test/english_query100.txt"
    #name = "20170407_res_labeled_0"
    #name = "20170407_res_labeled_1"
    #name = "20170407_res_labeled_2"

    # get graph configuration
    runtime_names = [
        #"cvae2-merge-stc-weibo",
        #"attn-s2s-merge-stc-weibo-downsample",
        #"vae-merge-stc-weibo",
        #"attn-s2s-all-downsample-addmem",
        #"attn-bi-s2s-all-downsample-addmem",
        #"attn-bi-s2s-all-downsample-addmem2",
        "attn-s2s-all-downsample-n-gram-addmem"
        #"cvae-512-noprior-noattn",
        #"cvae-1024-prior-attn",
        #"cvae-1024-prior-attn-addmem",
        #"cvae-512-noprior-attn",
        #"vae-1024-attn-addmem"
        #"cvae-512-prior-attn",
        #"cvae-512-prior-noattn",
        #"cvae-128-prior-attn"
    ]
    scorer_names = [ 
        #"stc-2-interact-qpr-negpr"
        #"stc-2-interact-qppr"
        #"stc-2-interact-qppr-2"
        #"stc-2-interact-qpr-negpr-shuf"
        #"stc-2-interact-qpr-negpr-obj0"
        #"attn-s2s-merge-stc-weibo-downsample",
        #"attn-s2s-all-downsample-addmem",
        #"attn-bi-s2s-all-downsample-addmem2",
        #"attn-s2s-all-downsample-n-gram-addmem"
    ]

    records = []
    orig_records = []
    with codecs.open(name, "r") as f:
        for line in f:
            line = line.strip()
            orig_records.append(line)
            line = re.sub(" +", "，", line)
            line = re.split("\t", line)[0]
            records.append(line)

    runtimes = {}
    scorer = {} 
    for model_name in runtime_names:
        ckpt_dir = os.path.join(train_root, model_name) 
        model = create(model_name, job_type="single", task_id=0)
        model.initialize_infer(train_root, gpu="0")
        runtimes[model_name] = model
        tf.reset_default_graph()

    #for score
    for model_name in scorer_names:
        ckpt_dir = os.path.join(train_root, model_name) 
        model = create(model_name, job_type="single", task_id=0)
        model.conf.for_score = True
        model.initialize_infer(train_root, gpu="2", variants="score")
        scorer[model_name] = model
        tf.reset_default_graph()

    command = sys.argv[1]
    batch_size = int(sys.argv[2])
    is_ch = True 
    for i in range(0, len(records), batch_size):
        batch_res_of_all_models = []
        batch = records[i:i+batch_size]
        for name in runtime_names:
            model = runtimes[name]
            input_feed = model.preproc(batch, use_seg=is_ch, for_deploy=True) 
            step_out = model.step(input_feed=input_feed, forward_only=True, debug=False, for_deploy=True)
            out, probs, attn = model.after_proc(step_out) 

            # Model scoring
            for k in range(len(out)):
                examples = ["%s\t%s" % (batch[k], " ".join(each)) for each in out[k]]
                for score_model_name in scorer:
                    if score_model_name == name:
                        continue
                    input_feed = scorer[score_model_name].preproc(examples, use_seg=is_ch, for_deploy=False)
                    step_out = scorer[score_model_name].step(input_feed=input_feed, forward_only=True, debug=False, for_deploy=True)
                    for n, prob in enumerate(step_out["logprobs"]):
                        probs[k][n] += prob

            # Nick's re-score and some interference
            batch_res = [] 
            for k in range(len(out)):
                batch_res.append(Nick_plan.score_with_prob_attn(name, out[k], probs[k], attn[k], alpha=0.8, beta=0.2, average_across_len=True))

            batch_res_of_all_models.append(batch_res)

        for n in range(len(batch_res_of_all_models[0])):
            one_res_of_all_model = []
            for m in range(len(batch_res_of_all_models)):
                #sorted_res = Nick_plan.rank(batch_res_of_all_models[m][n])
                sorted_res = batch_res_of_all_models[m][n]
                #print sorted_res[0][3].keys()
                #for each in sorted_res:
                #    print each[0], each[1], each[2], "[All:%.5f]" % each[3], "[Org:%.5f]" % each[4], "[LenRatio:%.5f]" % each[5], "[AttnEnt:%.5f]" % each[6]
                #    #print each[0], each[1], each[2], each[3].values()
                one_res_of_all_model.extend(sorted_res)
                #raw_input()

            final_res = sorted(one_res_of_all_model, key=lambda x:x[3], reverse=True)
            #final_res = sorted(one_res_of_all_model, key=lambda x:sum(x[3].values()) /float(len(x[1].decode("utf-8"))), reverse=True)
            if command == "see":
                print "==================================="
                print records[n + i]
                print "==================================="
                for each in final_res[0:80]:
                    print each[0], each[-1], each[2], "[All:%.5f]" % each[3], "[Org:%.5f]" % each[4], "[LenRatio:%.5f]" % each[5], "[AttnEnt:%.5f]" % each[6]
                #    print each[0], each[1], each[2], each[3]
                raw_input()
            elif command == "dump":
                print "%s\t%s\t%s" % (records[n + i], final_res[0][1], final_res[0][0])
            elif command == "dumpall":
                for w in range(len(final_res)):
                    print "%s\t%s\t%s" % (orig_records[n + i], final_res[w][1], final_res[w][0])
                #raw_input()
            else:
                print "command dump or see"

if __name__ == "__main__": 
    main()
