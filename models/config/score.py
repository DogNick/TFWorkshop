import os
import copy
from Config import confs
from Config import Config

confs["cvae-noattn-opensubtitle_gt3-emb"] = copy.deepcopy(confs["vae"])
confs["cvae-noattn-opensubtitle_gt3-emb"].model_kind           = "CVAERNNemb"
confs["cvae-noattn-opensubtitle_gt3-emb"].input_vocab_size     = 20000
confs["cvae-noattn-opensubtitle_gt3-emb"].output_vocab_size    = 20000
confs["cvae-noattn-opensubtitle_gt3-emb"].lr_check_steps       = 150 
confs["cvae-noattn-opensubtitle_gt3-emb"].lr_keep_steps        = 80000  
confs["cvae-noattn-opensubtitle_gt3-emb"].embedding_size       = 200
confs["cvae-noattn-opensubtitle_gt3-emb"].learning_rate        = 0.00001
confs["cvae-noattn-opensubtitle_gt3-emb"].learning_rate_decay_factor = 1.0 
confs["cvae-noattn-opensubtitle_gt3-emb"].batch_size           = 128 
confs["cvae-noattn-opensubtitle_gt3-emb"].cell_model           = "LSTMCell"
confs["cvae-noattn-opensubtitle_gt3-emb"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/opensubtitle_gt3"
confs["cvae-noattn-opensubtitle_gt3-emb"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/opensubtitle_gt3/opensubtitle_filter2k.w2v.npy"
confs["cvae-noattn-opensubtitle_gt3-emb"].input_max_len        = 35 
confs["cvae-noattn-opensubtitle_gt3-emb"].restore_from         = "cvae-noattn-opensubtitle_gt3" 
