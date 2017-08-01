import os
import copy
from Config import confs
from Config import Config

# Detailed Configuration of model 
##################################################################
confs["attn-s2s"] = Config()
confs["attn-s2s"].model_kind                    = "AttnSeq2Seq"
confs["attn-s2s"].learning_rate_decay_factor    = 1.0 
confs["attn-s2s"].learning_rate                 = 0.0001
confs["attn-s2s"].input_vocab_size              = 60000
confs["attn-s2s"].output_vocab_size             = 60000
confs["attn-s2s"].batch_size                    = 128 
confs["attn-s2s"].opt_name                      = "Adam" 
confs["attn-s2s"].max_to_keep                   = 10
confs["attn-s2s"].input_max_len                 = 35
confs["attn-s2s"].output_max_len                = 35
confs["attn-s2s"].num_layers                    = 4 
confs["attn-s2s"].num_units                     = 1024
confs["attn-s2s"].out_layer_size                = None
confs["attn-s2s"].embedding_init                = None 
confs["attn-s2s"].embedding_size                = 150
confs["attn-s2s"].cell_model                    = "GRUCell"
confs["attn-s2s"].output_keep_prob              = 1.0 
confs["attn-s2s"].num_samples                   = 512 
confs["attn-s2s"].enc_reverse                   = False 
confs["attn-s2s"].bidirectional                 = False 
confs["attn-s2s"].beam_splits                   = [1] 
confs["attn-s2s"].tokenize_mode                 = "word" 
############################################################


confs["attn-s2s-distributed"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-s2s-distributed"].batch_size = 1000 
confs["attn-s2s-distributed"].cluster = {
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
            "0.0.0.0:4004",
            "0.0.0.0:4005",
            #"0.0.0.0:4006",
            #"0.0.0.0:4007",
            #"0.0.0.0:4008",
            #"0.0.0.0:4009",
            #"0.0.0.0:40010",
            #"0.0.0.0:40011"
        ] 
}
confs["attn-s2s-distributed"].replicas_to_aggregate = 6 
confs["attn-s2s-distributed"].total_num_replicas = 6

confs["attn-s2s-topic"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-s2s-topic"].model_kind           = "DynAttnTopicSeq2Seq" 
confs["attn-s2s-topic"].num_units            = 1024 
confs["attn-s2s-topic"].output_keep_prob     = 0.9 
confs["attn-s2s-topic"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/STC-2-topic"
confs["attn-s2s-topic"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/STC-2-topic/weibo.w2v.npy"
confs["attn-s2s-topic"].topic_embedding_init = "/search/odin/Nick/GenerateWorkshop/data/STC-2-topic/topic_word_embeddings.npy"
confs["attn-s2s-topic"].topic_embedding_size = 200 
confs["attn-s2s-topic"].topic_vocab_size     = 78047 
confs["attn-s2s-topic"].reverse              = False 


##########################################################################
#
# Currently for stc-2 
#
##############################################################################################
confs["attn-s2s-all-downsample-n-gram-addmem"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-s2s-all-downsample-n-gram-addmem"].model_kind                    = "AttnSeq2Seq"
confs["attn-s2s-all-downsample-n-gram-addmem"].input_vocab_size              = 41000 
confs["attn-s2s-all-downsample-n-gram-addmem"].learning_rate                 = 0.0001
confs["attn-s2s-all-downsample-n-gram-addmem"].num_layers                    = 3 
confs["attn-s2s-all-downsample-n-gram-addmem"].output_vocab_size             = 41000
confs["attn-s2s-all-downsample-n-gram-addmem"].embedding_size                = 200 
confs["attn-s2s-all-downsample-n-gram-addmem"].cell_model                    = "GRUCell"
confs["attn-s2s-all-downsample-n-gram-addmem"].num_units                     = 2048 
confs["attn-s2s-all-downsample-n-gram-addmem"].addmem                        = True 
confs["attn-s2s-all-downsample-n-gram-addmem"].bidirectional                 = False 
confs["attn-s2s-all-downsample-n-gram-addmem"].embedding_init                = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-bought/weibo-stc-bought.w2v.npy" 
confs["attn-s2s-all-downsample-n-gram-addmem"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-bought"
confs["attn-s2s-all-downsample-n-gram-addmem"].beam_splits                   = [15,15,15,15,15,15,15]#[5,5,5,1,1,1,1]#[3,3,3,1,1,1] 
confs["attn-s2s-all-downsample-n-gram-addmem"].use_data_queue                = False 
confs["attn-s2s-all-downsample-n-gram-addmem"].output_keep_prob              = 1.0 

confs["attn-s2s-merge-stc-weibo-downsample"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-s2s-merge-stc-weibo-downsample"].model_kind                    = "AttnSeq2Seq"
confs["attn-s2s-merge-stc-weibo-downsample"].learning_rate                 = 0.00005
confs["attn-s2s-merge-stc-weibo-downsample"].input_vocab_size              = 40000 
confs["attn-s2s-merge-stc-weibo-downsample"].output_vocab_size             = 40000
confs["attn-s2s-merge-stc-weibo-downsample"].embedding_size                = 150 
confs["attn-s2s-merge-stc-weibo-downsample"].num_layers                    = 6 
confs["attn-s2s-merge-stc-weibo-downsample"].cell_model                    = "LSTMCell"
confs["attn-s2s-merge-stc-weibo-downsample"].batch_size                    = 128 
confs["attn-s2s-merge-stc-weibo-downsample"].embedding_init                = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["attn-s2s-merge-stc-weibo-downsample"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["attn-s2s-merge-stc-weibo-downsample"].beam_splits                   = [12,12,12,12,12,12,12,12] 
confs["attn-s2s-merge-stc-weibo-downsample"].use_data_queue                = False 

confs["attn-s2s-all-downsample-addmem"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-s2s-all-downsample-addmem"].model_kind                    = "AttnSeq2Seq"
confs["attn-s2s-all-downsample-addmem"].input_vocab_size              = 40000 
confs["attn-s2s-all-downsample-addmem"].learning_rate                 = 0.0001
confs["attn-s2s-all-downsample-addmem"].output_vocab_size             = 40000
confs["attn-s2s-all-downsample-addmem"].embedding_size                = 150 
confs["attn-s2s-all-downsample-addmem"].cell_model                    = "LSTMCell"
confs["attn-s2s-all-downsample-addmem"].addmem                        = True 
confs["attn-s2s-all-downsample-addmem"].embedding_init                = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["attn-s2s-all-downsample-addmem"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["attn-s2s-all-downsample-addmem"].beam_splits                   = [12,12,12,12,12,12,12,12]
confs["attn-s2s-all-downsample-addmem"].input_max_len                 = 35 
confs["attn-s2s-all-downsample-addmem"].output_max_len                = 35 
confs["attn-s2s-all-downsample-addmem"].use_data_queue                = False 

confs["attn-bi-s2s-all-downsample-addmem"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-bi-s2s-all-downsample-addmem"].model_kind                    = "AttnSeq2Seq"
confs["attn-bi-s2s-all-downsample-addmem"].input_vocab_size              = 40000 
confs["attn-bi-s2s-all-downsample-addmem"].learning_rate                 = 0.0001
confs["attn-bi-s2s-all-downsample-addmem"].output_vocab_size             = 40000
confs["attn-bi-s2s-all-downsample-addmem"].embedding_size                = 150 
confs["attn-bi-s2s-all-downsample-addmem"].cell_model                    = "LSTMCell"
confs["attn-bi-s2s-all-downsample-addmem"].num_units                     = 512 
confs["attn-bi-s2s-all-downsample-addmem"].addmem                        = True 
confs["attn-bi-s2s-all-downsample-addmem"].bidirectional                 = True 
confs["attn-bi-s2s-all-downsample-addmem"].embedding_init                = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["attn-bi-s2s-all-downsample-addmem"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["attn-bi-s2s-all-downsample-addmem"].beam_splits                   = [12,12,12,12,12,12,12,12,12,12]#[5,5,5,1,1,1,1]#[3,3,3,1,1,1] 
confs["attn-bi-s2s-all-downsample-addmem"].use_data_queue                = False 

confs["attn-bi-s2s-all-downsample-addmem2"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-bi-s2s-all-downsample-addmem2"].model_kind                    = "AttnSeq2Seq"
confs["attn-bi-s2s-all-downsample-addmem2"].input_vocab_size              = 40000 
confs["attn-bi-s2s-all-downsample-addmem2"].learning_rate                 = 0.0001
confs["attn-bi-s2s-all-downsample-addmem2"].output_vocab_size             = 41000
confs["attn-bi-s2s-all-downsample-addmem2"].embedding_size                = 200 
confs["attn-bi-s2s-all-downsample-addmem2"].cell_model                    = "GRUCell"
confs["attn-bi-s2s-all-downsample-addmem2"].num_units                     = 1024 
confs["attn-bi-s2s-all-downsample-addmem2"].addmem                        = True 
confs["attn-bi-s2s-all-downsample-addmem2"].bidirectional                 = True 
confs["attn-bi-s2s-all-downsample-addmem2"].embedding_init                = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-bought/weibo-stc-bought.w2v.npy" 
confs["attn-bi-s2s-all-downsample-addmem2"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-bought"
confs["attn-bi-s2s-all-downsample-addmem2"].beam_splits                   = [12,12,12,12,12,12,12,12,12,12]#[5,5,5,1,1,1,1]#[3,3,3,1,1,1] 
confs["attn-bi-s2s-all-downsample-addmem2"].use_data_queue                = False

# match poem
confs["attn-bi-poem"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-bi-poem"].model_kind                    = "AttnSeq2Seq"
confs["attn-bi-poem"].input_vocab_size              = 11000 
confs["attn-bi-poem"].output_vocab_size             = 11000
confs["attn-bi-poem"].num_layers 					= 3 
confs["attn-bi-poem"].input_max_len 				= 10 
confs["attn-bi-poem"].output_keep_prob              = 0.8
confs["attn-bi-poem"].num_units                     = 1024 
confs["attn-bi-poem"].data_dir                      = "/search/odin/offline/Workshop/data/poem"
confs["attn-bi-poem"].cell_model                    = "LSTMCell"
confs["attn-bi-poem"].opt_name                      = "COCOB" 
confs["attn-bi-poem"].embedding_size                = 150
confs["attn-bi-poem"].beam_splits                   = [6,6,6,6,6,6,6,6]#[2,2,2,2,2,2,2,2,2]#[1,1,3,3,2,1,1,1,1]#[2,2,2,1,1,1] 
confs["attn-bi-poem"].bidirectional                 = True 
confs["attn-bi-poem"].use_data_queue						 = False  

confs["attn-bi-poem-no-ave-len"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-bi-poem-no-ave-len"].model_kind                    = "AttnSeq2Seq"
confs["attn-bi-poem-no-ave-len"].input_vocab_size              = 11000 
confs["attn-bi-poem-no-ave-len"].output_vocab_size             = 11000
confs["attn-bi-poem-no-ave-len"].num_layers 				   = 3 
confs["attn-bi-poem-no-ave-len"].input_max_len 				   = 10 
confs["attn-bi-poem-no-ave-len"].output_keep_prob              = 0.8
confs["attn-bi-poem-no-ave-len"].num_units                     = 1024 
confs["attn-bi-poem-no-ave-len"].data_dir                      = "/search/odin/offline/Workshop/data/poem"
confs["attn-bi-poem-no-ave-len"].cell_model                    = "LSTMCell"
confs["attn-bi-poem-no-ave-len"].opt_name                      = "COCOB" 
confs["attn-bi-poem-no-ave-len"].embedding_size                = 150
confs["attn-bi-poem-no-ave-len"].beam_splits                   = [6,6,6,6,6,6,6,6]#[2,2,2,2,2,2,2,2,2]#[1,1,3,3,2,1,1,1,1]#[2,2,2,1,1,1] 
confs["attn-bi-poem-no-ave-len"].bidirectional                 = True 
confs["attn-bi-poem-no-ave-len"].use_data_queue						 = False 

confs["attn-bi-s2s-addmem-poem"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-bi-s2s-addmem-poem"].model_kind                    = "AttnSeq2Seq"
confs["attn-bi-s2s-addmem-poem"].input_vocab_size              = 12000 
confs["attn-bi-s2s-addmem-poem"].output_vocab_size             = 12000 

confs["attn-bi-s2s-addmem-poem"].learning_rate                 = 0.0001
confs["attn-bi-s2s-addmem-poem"].num_layers                    = 3 

confs["attn-bi-s2s-addmem-poem"].embedding_size                = 150 
confs["attn-bi-s2s-addmem-poem"].cell_model                    = "LSTMCell"
confs["attn-bi-s2s-addmem-poem"].num_units                     = 768 
confs["attn-bi-s2s-addmem-poem"].addmem                        = True 
confs["attn-bi-s2s-addmem-poem"].bidirectional                 = True 
confs["attn-bi-s2s-addmem-poem"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/poem"
confs["attn-bi-s2s-addmem-poem"].beam_splits                   = [30,30,30,30,30,30,30,30,30]#[12,12,12,12,12,12,12]#[5,5,5,1,1,1,1]#[3,3,3,1,1,1] 
confs["attn-bi-s2s-addmem-poem"].use_data_queue                = False 
confs["attn-bi-s2s-addmem-poem"].input_max_len                 = 10
confs["attn-bi-s2s-addmem-poem"].output_max_len                = 10 

confs["attn-bi-s2s-addmem-poem2"] = copy.deepcopy(confs["attn-s2s"])
confs["attn-bi-s2s-addmem-poem2"].model_kind                    = "AttnSeq2Seq"
confs["attn-bi-s2s-addmem-poem2"].input_vocab_size              = 12000 
confs["attn-bi-s2s-addmem-poem2"].output_vocab_size             = 12000 

confs["attn-bi-s2s-addmem-poem2"].learning_rate                 = 0.0001
confs["attn-bi-s2s-addmem-poem2"].num_layers                    = 3 

confs["attn-bi-s2s-addmem-poem2"].embedding_size                = 150 
confs["attn-bi-s2s-addmem-poem2"].cell_model                    = "GRUCell"
confs["attn-bi-s2s-addmem-poem2"].num_units                     = 1024 
confs["attn-bi-s2s-addmem-poem2"].addmem                        = True 
confs["attn-bi-s2s-addmem-poem2"].bidirectional                 = True 
confs["attn-bi-s2s-addmem-poem2"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/poem"
confs["attn-bi-s2s-addmem-poem2"].beam_splits                   = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[12,12,12,12,12,12,12]#[5,5,5,1,1,1,1]#[3,3,3,1,1,1] 
confs["attn-bi-s2s-addmem-poem2"].use_data_queue                = False 
confs["attn-bi-s2s-addmem-poem2"].input_max_len                 = 10
confs["attn-bi-s2s-addmem-poem2"].output_max_len                = 10 


# Judge poem
confs["rnncls-bi-judge_poem"] = Config() 
confs["rnncls-bi-judge_poem"].model_kind							 = "RNNClassification"  
confs["rnncls-bi-judge_poem"].input_vocab_size                       = 11000  
confs["rnncls-bi-judge_poem"].output_vocab_size                       = 11000  
confs["rnncls-bi-judge_poem"].num_layers 							 = 3  
confs["rnncls-bi-judge_poem"].embedding_size                         = 150 
confs["rnncls-bi-judge_poem"].bidirectional							 = True 
confs["rnncls-bi-judge_poem"].data_dir								 = "/search/odin/offline/Workshop/data/judge_poem" 
confs["rnncls-bi-judge_poem"].embedding_init       					 = None 
confs["rnncls-bi-judge_poem"].use_data_queue						 = False  
confs["rnncls-bi-judge_poem"].input_max_len 						 = 35 
confs["rnncls-bi-judge_poem"].tokenize_mode      		             = "word" 
confs["rnncls-bi-judge_poem"].batch_size         		             = 128 
confs["rnncls-bi-judge_poem"].num_units          		             = 1024 
confs["rnncls-bi-judge_poem"].cell_model         		             = "GRUCell"
confs["rnncls-bi-judge_poem"].opt_name      		                 = "COCOB" 
#confs["rnncls-bi-judge_poem"].learning_rate      		             = 0.01 
confs["rnncls-bi-judge_poem"].max_to_keep      		                 = 10 

confs["rnncls-bi-judge_poem"].conv_conf = [
                        			[(1, 2048, 1, 128), (1, 2048), (2, 2), (2, 2)],
                        			[(2, 1, 128, 64), (2, 1), (2, 2), (2, 2)]
							  ]
confs["rnncls-bi-judge_poem"].fc_h_size = 100 
confs["rnncls-bi-judge_poem"].tag_num = 2 
