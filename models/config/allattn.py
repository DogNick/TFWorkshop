import os
import copy
from Config import confs, Config

confs["allattn"] = Config()
confs["allattn"].model_kind						= "AllAttn" 
confs["allattn"].learning_rate					= 0.00004
confs["allattn"].input_vocab_size				= 41000
confs["allattn"].output_vocab_size				= 41000
confs["allattn"].batch_size			            = 128 
confs["allattn"].input_max_len			        = 30 
confs["allattn"].output_max_len                 = 30 

confs["allattn"].hidden_units					= 512
confs["allattn"].num_heads						= 8 
confs["allattn"].num_blocks					    = 6 
confs["allattn"].max_to_keep					= 10 
confs["allattn"].embedding_size					= 512 
confs["allattn"].embedding_init                 = None #"/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy"
confs["allattn"].data_dir						= "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean" 
confs["allattn"].use_data_queue                 = False 
confs["allattn"].tokenize_mode                  = "word" 
confs["allattn"].dropout_rate                   = 0.1 
confs["allattn"].opt_name                       = "Adam" 


confs["allattn_dist"] = copy.deepcopy(confs["allattn"])
confs["allattn_dist"].use_data_queue			= True 
confs["allattn_dist"].replicas_to_aggregate		= 4 
confs["allattn_dist"].batch_size				= 512 
confs["allattn_dist"].total_num_replicas		= 4 
confs["allattn_dist"].cluster                   = {
                                                  "ps": [
                                                      "0.0.0.0:3990",
                                                      "0.0.0.0:3991",
                                                      "0.0.0.0:3992",
                                                      #"0.0.0.0:3993",
                                                  ],
                                                  "worker": [
                                                      "0.0.0.0:4000",
                                                      "0.0.0.0:4001",
                                                      "0.0.0.0:4002",
                                                      "0.0.0.0:4003",
                                                      #"0.0.0.0:4004",
                                                      #"0.0.0.0:4005",
                                                      #"0.0.0.0:4006",
                                                      #"0.0.0.0:4007",
                                                  ] 
											   }

