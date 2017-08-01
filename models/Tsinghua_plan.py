import tensorflow as tf
from util import *
from config import *
import time
class Tsinghua(object):
    def __init__(self, name, for_deploy, job_type="single", dtype=tf.float32, data_dequeue_op=None):
        self.name = name
        self.conf = conf = confs[name]
        self.model_kind = self.__class__.__name__
        if self.conf.model_kind !=  self.model_kind:
            print "Wrong model kind !, this model needs config of kind '%s', but a '%s' config is given." % (self.model_kind, self.conf.model_kind)
            exit(0)

    def fetch_test_data(self, records, begin=0, size=128, is_ch=True):
        data = []
        for each in records[begin:begin + size]:
            data.append([each, len(each) + 1])
        return data

    def get_batch(self, examples):
        conf = self.conf
        batch_enc_inps,  batch_enc_lens = [], []
        for encs, enc_len in examples:
            enc_len = enc_len if enc_len < conf.input_max_len else conf.input_max_len
            encs = encs[0:conf.input_max_len]
            if conf.enc_reverse:
                encs = list(reversed(encs + ["_PAD"] * (enc_len - len(encs))))
            enc_inps = encs + ["_PAD"] * (conf.input_max_len - len(encs))

            batch_enc_inps.append(enc_inps)
            batch_enc_lens.append(np.int32(enc_len))

            feed_dict = {
                    "enc_inps:0": batch_enc_inps,
                    "enc_lens:0": batch_enc_lens
            } 
        return feed_dict 

    def after_proc(self, out): 
        t0 = time.time()
        parents =  out["beam_parents"]
        symbols = out["beam_symbols"]
        result_parents = out["result_parents"]
        result_symbols = out["result_symbols"]
        result_probs = out["result_probs"]

        res = []
        for batch, (prbs, smbs, prts) in enumerate(zip(result_probs, result_symbols, result_parents)):
            _res = []
            symbol = symbols[batch]
            parent = parents[batch]
            for i, (prb, smb, prt) in enumerate(zip(prbs, smbs, prts)):
                end = []
                for idx, j in enumerate(smb):
                    if j == '_EOS':
                        end.append(idx)
                if len(end) == 0: continue
                for j in end:
                    p = prt[j]
                    s = -1
                    output = []
                    for step in xrange(i-1, -1, -1):
                        s = symbol[step][p]
                        p = parent[step][p]
                        output.append(s)
                    output.reverse()
                    res.append([output, -prb[j]/(len(output))])
        print "beam_time: %.6f" % (time.time() - t0)
        return res 

class Postprob(object):
    def __init__(self, name, for_deploy, job_type="single", dtype=tf.float32, data_dequeue_op=None):
        self.name = name
        self.conf = conf = confs[name]
        self.model_kind = self.__class__.__name__
        if self.conf.model_kind !=  self.model_kind:
            print "Wrong model kind !, this model needs config of kind '%s', but a '%s' config is given." % (self.model_kind, self.conf.model_kind)
            exit(0)

    def fetch_test_data(self, records, begin=0, size=128, is_ch=True):
        data = []
        # reverse post and response here
        for p, r in records[begin:begin + size]:
            data.append([r, len(r) + 1, p, len(p) + 1])
        return data

    def get_batch(self, examples):
        conf = self.conf
        batch_enc_inps, batch_dec_inps, batch_enc_lens, batch_dec_lens = [], [], [], []
        for encs, enc_len, decs, dec_len in examples:
            enc_len = enc_len if enc_len < conf.input_max_len else conf.input_max_len
            encs = encs[0:conf.input_max_len]
            enc_inps = encs + ["_PAD"] * (conf.input_max_len - len(encs))
            batch_enc_inps.append(enc_inps)
            batch_enc_lens.append(np.int32(enc_len))
            decs += ["_EOS"]
            decs = decs[0:conf.output_max_len + 1]
            # fit to the max_dec_len
            if dec_len > conf.output_max_len + 1:
                dec_len = conf.output_max_len + 1

            # Merge dec inps and targets 
            batch_dec_inps.append(decs + ["_PAD"] * (conf.output_max_len + 1 - len(decs)))
            batch_dec_lens.append(np.int32(dec_len))

        max_len = max(batch_dec_lens)
        batch_dec_inps = [each[0:max_len] for each in batch_dec_inps]
        feed_dict = {
                "enc_inps:0": batch_enc_inps,
                "enc_lens:0": batch_enc_lens,
                "dec_inps:0": batch_dec_inps,
                "dec_lens:0": batch_dec_lens
        }
        return feed_dict 

    def after_proc(self, out): 
        return out["batch_loss"] 

def postprob_rerank(post, cans):
    sorted_cans = [can[0] for can in cans]
    return sorted_cans 
