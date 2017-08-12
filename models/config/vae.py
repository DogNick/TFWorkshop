import os
import copy
from Config import confs
from Config import Config

confs["vae"] = Config()
confs["vae"].model_kind                    = "VAERNN"
confs["vae"].learning_rate_decay_factor    = 1.0 
confs["vae"].learning_rate                 = 0.0001
confs["vae"].input_vocab_size              = 60000
confs["vae"].output_vocab_size             = 60000
confs["vae"].batch_size                    = 128 
confs["vae"].opt_name                      = "Adam" 
confs["vae"].max_to_keep                   = 10
confs["vae"].input_max_len                 = 35
confs["vae"].output_max_len                = 35
confs["vae"].num_layers                    = 4 
confs["vae"].num_units                     = 1024
confs["vae"].out_layer_size                = None
confs["vae"].embedding_init                = None 
confs["vae"].embedding_size                = 150
confs["vae"].cell_model                    = "GRUCell"
confs["vae"].output_keep_prob              = 1.0 
confs["vae"].num_samples                   = 512 
confs["vae"].enc_reverse                   = False 
confs["vae"].bidirectional                 = False 
confs["vae"].beam_splits                   = [1] 
confs["vae"].tokenize_mode                 = "word" 


#########################################
confs["vae2-opensubtitle"] = copy.deepcopy(confs["vae"])
confs["vae2-opensubtitle"].model_kind           = "VAERNN2"
confs["vae2-opensubtitle"].input_vocab_size     = 30000
confs["vae2-opensubtitle"].output_vocab_size    = 30000 
confs["vae2-opensubtitle"].lr_check_steps       = 150 
confs["vae2-opensubtitle"].lr_keep_steps        = 80000  
confs["vae2-opensubtitle"].learning_rate        = 0.0001
confs["vae2-opensubtitle"].num_layers        	= 3 
confs["vae2-opensubtitle"].learning_rate_decay_factor = 1.0 
confs["vae2-opensubtitle"].batch_size           = 128 
confs["vae2-opensubtitle"].cell_model           = "LSTMCell"
#confs["vae2-opensubtitle"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["vae2-opensubtitle"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/opensubtitle"
confs["vae2-opensubtitle"].addmem               = True 
confs["vae2-opensubtitle"].input_max_len        = 35
confs["vae2-opensubtitle"].output_max_len       = 35
confs["vae2-opensubtitle"].opt_name             = "Adam" 
#confs["vae2-opensubtitle"].opt_name             = "SGD" 
#confs["vae2-opensubtitle"].opt_name             = "COCOB" 
confs["vae2-opensubtitle"].enc_latent_dim       = 256 
confs["vae2-opensubtitle"].lam                  = 0.0 
confs["vae2-opensubtitle"].beam_splits          = [8,8,8,8,8,8,8,8,8] 
confs["vae2-opensubtitle"].use_data_queue       = False
confs["vae2-opensubtitle"].stddev               = 1.0 
confs["vae2-opensubtitle"].kld_ratio            = 0.01 



#######################################
confs["vae-merge-stc-weibo"] = copy.deepcopy(confs["vae"])
confs["vae-merge-stc-weibo"].model_kind           = "VAERNN"
confs["vae-merge-stc-weibo"].input_vocab_size     = 40000
confs["vae-merge-stc-weibo"].output_vocab_size    = 40000
confs["vae-merge-stc-weibo"].lr_check_steps       = 150 
confs["vae-merge-stc-weibo"].lr_keep_steps        = 80000  
confs["vae-merge-stc-weibo"].learning_rate        = 0.00002
confs["vae-merge-stc-weibo"].learning_rate_decay_factor = 1.0 
confs["vae-merge-stc-weibo"].batch_size           = 128 
confs["vae-merge-stc-weibo"].cell_model           = "LSTMCell"
confs["vae-merge-stc-weibo"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["vae-merge-stc-weibo"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["vae-merge-stc-weibo"].input_max_len        = 35 
confs["vae-merge-stc-weibo"].output_max_len       = 35 
confs["vae-merge-stc-weibo"].opt_name             = "Adam" 
confs["vae-merge-stc-weibo"].enc_latent_dim       = 1024 
confs["vae-merge-stc-weibo"].lam                  = 0.0 
confs["vae-merge-stc-weibo"].beam_splits          = [8,8,8,8,8,8,8,8,8] 
confs["vae-merge-stc-weibo"].use_data_queue       = False
confs["vae-merge-stc-weibo"].stddev               = 3.0 


confs["cvae-simpleprior-reddit-addmem"] = copy.deepcopy(confs["vae"])
confs["cvae-simpleprior-reddit-addmem"].model_kind                    = "CVAERNN"
confs["cvae-simpleprior-reddit-addmem"].batch_size                    = 128 
confs["cvae-simpleprior-reddit-addmem"].input_vocab_size              = 30000
confs["cvae-simpleprior-reddit-addmem"].output_vocab_size             = 30000 

confs["cvae-simpleprior-reddit-addmem"].learning_rate                 = 0.0001
confs["cvae-simpleprior-reddit-addmem"].num_layers                    = 3 
confs["cvae-simpleprior-reddit-addmem"].embedding_size                = 200 
confs["cvae-simpleprior-reddit-addmem"].cell_model                    = "GRUCell"
confs["cvae-simpleprior-reddit-addmem"].num_units                     = 1024 
confs["cvae-simpleprior-reddit-addmem"].addmem                        = True 
#confs["cvae-simpleprior-reddit-addmem"].bidirectional                 = True 
confs["cvae-simpleprior-reddit-addmem"].embedding_init				  = "/search/odin/Nick/GenerateWorkshop/data/SRT-reddit-proced-voca-filtered/pair_proced3_for_w2v.w2v.npy"
confs["cvae-simpleprior-reddit-addmem"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/SRT-reddit-proced-voca-filtered"
confs["cvae-simpleprior-reddit-addmem"].beam_splits                   = [20,20,20,20,20,20,20,20]#[12,12,12,12,12,12,12]#[5,5,5,1,1,1,1]#[3,3,3,1,1,1] 
#confs["cvae-simpleprior-reddit-addmem"].use_data_queue                = False
confs["cvae-simpleprior-reddit-addmem"].use_data_queue                = False 
confs["cvae-simpleprior-reddit-addmem"].input_max_len                 = 25 
confs["cvae-simpleprior-reddit-addmem"].output_max_len                = 30 
confs["cvae-simpleprior-reddit-addmem"].prior_type                    = "simple"
confs["cvae-simpleprior-reddit-addmem"].enc_latent_dim                = 128 
confs["cvae-simpleprior-reddit-addmem"].stddev                        = 1.0
confs["cvae-simpleprior-reddit-addmem"].kld_ratio                     = 1.0
confs["cvae-simpleprior-reddit-addmem"].bow_ratio                     = 0.0 
confs["cvae-simpleprior-reddit-addmem"].opt_name					  = "Adam" 


confs["vae-reddit-addmem"] = copy.deepcopy(confs["vae"])
confs["vae-reddit-addmem"].model_kind           = "VAERNN"
confs["vae-reddit-addmem"].input_vocab_size     = 30000
confs["vae-reddit-addmem"].output_vocab_size    = 30000
confs["vae-reddit-addmem"].lr_check_steps       = 200 
confs["vae-reddit-addmem"].lr_keep_steps        = 80000  
confs["vae-reddit-addmem"].learning_rate        = 0.0001
confs["vae-reddit-addmem"].learning_rate_decay_factor = 1.0 
confs["vae-reddit-addmem"].batch_size           = 128 
confs["vae-reddit-addmem"].cell_model           = "GRUCell"

confs["vae-reddit-addmem"].embedding_size       = 200 
confs["vae-reddit-addmem"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/SRT-reddit-proced-voca-filtered/pair_proced3_for_w2v.w2v.npy"
confs["vae-reddit-addmem"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/SRT-reddit-proced-voca-filtered"
confs["vae-reddit-addmem"].input_max_len        = 35 
confs["vae-reddit-addmem"].output_max_len       = 40 
confs["vae-reddit-addmem"].opt_name             = "Adam" 
confs["vae-reddit-addmem"].enc_latent_dim       = 1024 
confs["vae-reddit-addmem"].lam                  = 0.0 
confs["vae-reddit-addmem"].beam_splits          = [8,8,8,8,8,8,8,8,8] 
confs["vae-reddit-addmem"].use_data_queue       = False
confs["vae-reddit-addmem"].stddev               = 1.0 
confs["vae-reddit-addmem"].addmem               = True 


##################################################################################
#
# New CVAE
####################################################################################################################################
confs["cvae-512-prior-noattn"] = copy.deepcopy(confs["vae"])
confs["cvae-512-prior-noattn"].model_kind           = "CVAERNN"
confs["cvae-512-prior-noattn"].input_vocab_size     = 40000
confs["cvae-512-prior-noattn"].output_vocab_size    = 40000
confs["cvae-512-prior-noattn"].lr_check_steps       = 150 
confs["cvae-512-prior-noattn"].lr_keep_steps        = 80000  
confs["cvae-512-prior-noattn"].learning_rate        = 0.0001
confs["cvae-512-prior-noattn"].learning_rate_decay_factor = 1.0 
confs["cvae-512-prior-noattn"].batch_size           = 128 
confs["cvae-512-prior-noattn"].cell_model           = "LSTMCell"
confs["cvae-512-prior-noattn"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["cvae-512-prior-noattn"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["cvae-512-prior-noattn"].input_max_len        = 35
confs["cvae-512-prior-noattn"].output_max_len       = 35 
confs["cvae-512-prior-noattn"].opt_name             = "Adam" 
confs["cvae-512-prior-noattn"].lam                  = 0.0 
confs["cvae-512-prior-noattn"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[10,10,10,10,10,10]#[2,2,2,2,2,2,2,2]#[10,10,10,10,10,10]#[4,4,4,1,1,1,1] 
confs["cvae-512-prior-noattn"].use_data_queue       = False

confs["cvae-512-prior-noattn"].enc_latent_dim       = 512 
confs["cvae-512-prior-noattn"].stddev               = 1.0 
confs["cvae-512-prior-noattn"].kld_ratio            = 1.0 
confs["cvae-512-prior-noattn"].bow_ratio            = 0.3
confs["cvae-512-prior-noattn"].prior_type           = "mlp" 
confs["cvae-512-prior-noattn"].attention            = None 

confs["cvae-512-noprior-noattn"] = copy.deepcopy(confs["vae"])
confs["cvae-512-noprior-noattn"].model_kind           = "CVAERNN"
confs["cvae-512-noprior-noattn"].input_vocab_size     = 40000
confs["cvae-512-noprior-noattn"].output_vocab_size    = 40000
confs["cvae-512-noprior-noattn"].lr_check_steps       = 150 
confs["cvae-512-noprior-noattn"].lr_keep_steps        = 80000  
confs["cvae-512-noprior-noattn"].learning_rate        = 0.0001
confs["cvae-512-noprior-noattn"].learning_rate_decay_factor = 1.0 
confs["cvae-512-noprior-noattn"].batch_size           = 128 
confs["cvae-512-noprior-noattn"].cell_model           = "LSTMCell"
confs["cvae-512-noprior-noattn"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["cvae-512-noprior-noattn"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["cvae-512-noprior-noattn"].input_max_len        = 35
confs["cvae-512-noprior-noattn"].output_max_len       = 35 
confs["cvae-512-noprior-noattn"].opt_name             = "Adam" 
confs["cvae-512-noprior-noattn"].lam                  = 0.0 
confs["cvae-512-noprior-noattn"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[10,10,10,10,10,10]#[2,2,2,2,2,2,2,2]#[10,10,10,10,10,10]#[4,4,4,1,1,1,1] 
confs["cvae-512-noprior-noattn"].use_data_queue       = False

confs["cvae-512-noprior-noattn"].enc_latent_dim       = 512 
confs["cvae-512-noprior-noattn"].stddev               = 1.0 
confs["cvae-512-noprior-noattn"].kld_ratio            = 1.0
confs["cvae-512-noprior-noattn"].bow_ratio            = 0.3
confs["cvae-512-noprior-noattn"].prior_type           = "" 
confs["cvae-512-noprior-noattn"].attention            = None

confs["cvae-512-prior-attn"] = copy.deepcopy(confs["vae"])
confs["cvae-512-prior-attn"].model_kind           = "CVAERNN"
confs["cvae-512-prior-attn"].input_vocab_size     = 40000
confs["cvae-512-prior-attn"].output_vocab_size    = 40000
confs["cvae-512-prior-attn"].lr_check_steps       = 150 
confs["cvae-512-prior-attn"].lr_keep_steps        = 80000  
confs["cvae-512-prior-attn"].learning_rate        = 0.0001
confs["cvae-512-prior-attn"].learning_rate_decay_factor = 1.0 
confs["cvae-512-prior-attn"].batch_size           = 128 
confs["cvae-512-prior-attn"].cell_model           = "LSTMCell"
confs["cvae-512-prior-attn"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["cvae-512-prior-attn"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["cvae-512-prior-attn"].input_max_len        = 35
confs["cvae-512-prior-attn"].output_max_len       = 35 
confs["cvae-512-prior-attn"].opt_name             = "Adam" 
confs["cvae-512-prior-attn"].lam                  = 0.0 
confs["cvae-512-prior-attn"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[2,2,2,2,2,2,2,2]#[10,10,10,10,10,10]#[4,4,4,1,1,1,1] 
confs["cvae-512-prior-attn"].use_data_queue       = False

confs["cvae-512-prior-attn"].enc_latent_dim       = 512 
confs["cvae-512-prior-attn"].stddev               = 1.0 
confs["cvae-512-prior-attn"].kld_ratio            = 1.0
confs["cvae-512-prior-attn"].bow_ratio            = 0.3
confs["cvae-512-prior-attn"].prior_type           = "mlp" 
confs["cvae-512-prior-attn"].attention            = "Luo" 

confs["cvae-512-noprior-attn"] = copy.deepcopy(confs["vae"])
confs["cvae-512-noprior-attn"].model_kind           = "CVAERNN"
confs["cvae-512-noprior-attn"].input_vocab_size     = 40000
confs["cvae-512-noprior-attn"].output_vocab_size    = 40000
confs["cvae-512-noprior-attn"].lr_check_steps       = 150 
confs["cvae-512-noprior-attn"].lr_keep_steps        = 80000  
confs["cvae-512-noprior-attn"].learning_rate        = 0.0001
confs["cvae-512-noprior-attn"].learning_rate_decay_factor = 1.0 
confs["cvae-512-noprior-attn"].batch_size           = 128 
confs["cvae-512-noprior-attn"].cell_model           = "LSTMCell"
confs["cvae-512-noprior-attn"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["cvae-512-noprior-attn"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["cvae-512-noprior-attn"].input_max_len        = 35
confs["cvae-512-noprior-attn"].output_max_len       = 35 
confs["cvae-512-noprior-attn"].opt_name             = "Adam" 
confs["cvae-512-noprior-attn"].lam                  = 0.0 
confs["cvae-512-noprior-attn"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[2,2,2,2,2,2,2,2]#[10,10,10,10,10,10]#[4,4,4,1,1,1,1] 
confs["cvae-512-noprior-attn"].use_data_queue       = False

confs["cvae-512-noprior-attn"].enc_latent_dim       = 512 
confs["cvae-512-noprior-attn"].stddev               = 1.0 
confs["cvae-512-noprior-attn"].kld_ratio            = 0.8
confs["cvae-512-noprior-attn"].bow_ratio            = 0.3
confs["cvae-512-noprior-attn"].prior_type           = "" 
confs["cvae-512-noprior-attn"].attention            = "Luo" 



confs["cvae-1024-simpleprior-attn"] = copy.deepcopy(confs["vae"])
confs["cvae-1024-simpleprior-attn"].model_kind           = "CVAERNN"
confs["cvae-1024-simpleprior-attn"].input_vocab_size     = 40000
confs["cvae-1024-simpleprior-attn"].output_vocab_size    = 40000
confs["cvae-1024-simpleprior-attn"].lr_check_steps       = 150 
confs["cvae-1024-simpleprior-attn"].lr_keep_steps        = 80000  
confs["cvae-1024-simpleprior-attn"].learning_rate        = 0.0
confs["cvae-1024-simpleprior-attn"].learning_rate_decay_factor = 1.0 
confs["cvae-1024-simpleprior-attn"].batch_size           = 128 
confs["cvae-1024-simpleprior-attn"].cell_model           = "LSTMCell"
confs["cvae-1024-simpleprior-attn"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["cvae-1024-simpleprior-attn"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["cvae-1024-simpleprior-attn"].input_max_len        = 35
confs["cvae-1024-simpleprior-attn"].output_max_len       = 35 
confs["cvae-1024-simpleprior-attn"].opt_name             = "COCOB" 
confs["cvae-1024-simpleprior-attn"].lam                  = 0.0 
confs["cvae-1024-simpleprior-attn"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[2,2,2,2,2,2,2,2]#[10,10,10,10,10,10]#[4,4,4,1,1,1,1] 
confs["cvae-1024-simpleprior-attn"].use_data_queue       = False

confs["cvae-1024-simpleprior-attn"].enc_latent_dim       = 1024 
confs["cvae-1024-simpleprior-attn"].stddev               = 1.0 
confs["cvae-1024-simpleprior-attn"].kld_ratio            = 1.0 
confs["cvae-1024-simpleprior-attn"].bow_ratio            = 0.0
confs["cvae-1024-simpleprior-attn"].prior_type           = "simple" 
confs["cvae-1024-simpleprior-attn"].attention            = "Luo"

confs["cvae-128-simpleprior-attn"] = copy.deepcopy(confs["vae"])
confs["cvae-128-simpleprior-attn"].model_kind           = "CVAERNN"
confs["cvae-128-simpleprior-attn"].input_vocab_size     = 40000
confs["cvae-128-simpleprior-attn"].output_vocab_size    = 40000
confs["cvae-128-simpleprior-attn"].lr_check_steps       = 150 
confs["cvae-128-simpleprior-attn"].lr_keep_steps        = 80000  
confs["cvae-128-simpleprior-attn"].learning_rate        = 0.0 
confs["cvae-128-simpleprior-attn"].learning_rate_decay_factor = 1.0 
confs["cvae-128-simpleprior-attn"].batch_size           = 128 
confs["cvae-128-simpleprior-attn"].cell_model           = "LSTMCell"
confs["cvae-128-simpleprior-attn"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["cvae-128-simpleprior-attn"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["cvae-128-simpleprior-attn"].input_max_len        = 35
confs["cvae-128-simpleprior-attn"].output_max_len       = 35 
confs["cvae-128-simpleprior-attn"].opt_name             = "COCOB" 
confs["cvae-128-simpleprior-attn"].lam                  = 0.0
confs["cvae-128-simpleprior-attn"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[10,10,10,10,10,10]#[2,2,2,2,2,2,2,2]#[10,10,10,10,10,10]#[4,4,4,1,1,1,1] 
confs["cvae-128-simpleprior-attn"].use_data_queue       = False
confs["cvae-128-simpleprior-attn"].enc_latent_dim       = 128 
confs["cvae-128-simpleprior-attn"].stddev               = 1.0 
confs["cvae-128-simpleprior-attn"].kld_ratio            = 1.0
confs["cvae-128-simpleprior-attn"].bow_ratio            = 0.0
confs["cvae-128-simpleprior-attn"].prior_type           = "simple" 
confs["cvae-128-simpleprior-attn"].attention            = "Luo"

confs["cvae-1024-simpleprior-attn-addmem"] = copy.deepcopy(confs["vae"])
confs["cvae-1024-simpleprior-attn-addmem"].model_kind           = "CVAERNN"
confs["cvae-1024-simpleprior-attn-addmem"].input_vocab_size     = 40000
confs["cvae-1024-simpleprior-attn-addmem"].output_vocab_size    = 40000
confs["cvae-1024-simpleprior-attn-addmem"].lr_check_steps       = 150 
confs["cvae-1024-simpleprior-attn-addmem"].lr_keep_steps        = 80000  
confs["cvae-1024-simpleprior-attn-addmem"].learning_rate        = 0.0 
confs["cvae-1024-simpleprior-attn-addmem"].learning_rate_decay_factor = 1.0 
confs["cvae-1024-simpleprior-attn-addmem"].batch_size           = 128 
confs["cvae-1024-simpleprior-attn-addmem"].cell_model           = "LSTMCell"
confs["cvae-1024-simpleprior-attn-addmem"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["cvae-1024-simpleprior-attn-addmem"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["cvae-1024-simpleprior-attn-addmem"].input_max_len        = 35
confs["cvae-1024-simpleprior-attn-addmem"].output_max_len       = 35 
confs["cvae-1024-simpleprior-attn-addmem"].opt_name             = "COCOB" 
confs["cvae-1024-simpleprior-attn-addmem"].lam                  = 0.0 
confs["cvae-1024-simpleprior-attn-addmem"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[10,10,10,10,10,10]#[2,2,2,2,2,2,2,2]#[10,10,10,10,10,10]#[4,4,4,1,1,1,1] 
confs["cvae-1024-simpleprior-attn-addmem"].use_data_queue       = False

confs["cvae-1024-simpleprior-attn-addmem"].enc_latent_dim       = 1024 
confs["cvae-1024-simpleprior-attn-addmem"].stddev               = 1.0 
confs["cvae-1024-simpleprior-attn-addmem"].kld_ratio            = 1.0
confs["cvae-1024-simpleprior-attn-addmem"].bow_ratio            = 0.0
confs["cvae-1024-simpleprior-attn-addmem"].prior_type           = "simple" 
confs["cvae-1024-simpleprior-attn-addmem"].attention            = "Luo"
confs["cvae-1024-simpleprior-attn-addmem"].addmem               = True 

confs["vae-1024-attn-addmem"] = copy.deepcopy(confs["vae"])
confs["vae-1024-attn-addmem"].model_kind           = "VAERNN"
confs["vae-1024-attn-addmem"].input_vocab_size     = 40000
confs["vae-1024-attn-addmem"].output_vocab_size    = 40000
confs["vae-1024-attn-addmem"].lr_check_steps       = 150 
confs["vae-1024-attn-addmem"].lr_keep_steps        = 80000  
confs["vae-1024-attn-addmem"].learning_rate        = 0.0001
confs["vae-1024-attn-addmem"].learning_rate_decay_factor = 1.0 
confs["vae-1024-attn-addmem"].batch_size           = 128 
confs["vae-1024-attn-addmem"].cell_model           = "LSTMCell"
#confs["vae-1024-attn-addmem"].embedding_init       = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-2-downsample/weibo.w2v.npy" 
confs["vae-1024-attn-addmem"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/WEIBO-stc-all-clean"
confs["vae-1024-attn-addmem"].input_max_len        = 35 
confs["vae-1024-attn-addmem"].output_max_len       = 35 
confs["vae-1024-attn-addmem"].opt_name             = "Adam" 
confs["vae-1024-attn-addmem"].enc_latent_dim       = 1024 
confs["vae-1024-attn-addmem"].lam                  = 0.0 
confs["vae-1024-attn-addmem"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[4,4,4,4,4,4,4,4,4,4]#[8,8,8,8,8,8,8,8,8]#[6,6,6,6,6,6,6,6]#test[2,2,2,2,2,2,2,2] 
confs["vae-1024-attn-addmem"].use_data_queue       = False
#confs["vae-1024-attn-addmem"].stddev               = 1.0 
confs["vae-1024-attn-addmem"].stddev               = 3.0 
confs["vae-1024-attn-addmem"].addmem               = True 

confs["vae-bi-1024-attn-addmem-poem"] = copy.deepcopy(confs["vae"])
confs["vae-bi-1024-attn-addmem-poem"].model_kind           = "VAERNN"
confs["vae-bi-1024-attn-addmem-poem"].input_vocab_size     = 12000
confs["vae-bi-1024-attn-addmem-poem"].output_vocab_size    = 12000
confs["vae-bi-1024-attn-addmem-poem"].lr_check_steps       = 150 
confs["vae-bi-1024-attn-addmem-poem"].num_layers           = 2 
confs["vae-bi-1024-attn-addmem-poem"].lr_keep_steps        = 80000  
confs["vae-bi-1024-attn-addmem-poem"].batch_size           = 128 
confs["vae-bi-1024-attn-addmem-poem"].cell_model           = "GRUCell"
confs["vae-bi-1024-attn-addmem-poem"].data_dir             = "/search/odin/Nick/GenerateWorkshop/data/poem"
confs["vae-bi-1024-attn-addmem-poem"].input_max_len        = 10 
confs["vae-bi-1024-attn-addmem-poem"].output_max_len       = 10 
confs["vae-bi-1024-attn-addmem-poem"].opt_name             = "COCOB" 
confs["vae-bi-1024-attn-addmem-poem"].enc_latent_dim       = 1024 
confs["vae-bi-1024-attn-addmem-poem"].lam                  = 0.0 
confs["vae-bi-1024-attn-addmem-poem"].beam_splits          = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]#[4,4,4,4,4,4,4,4,4,4]#[8,8,8,8,8,8,8,8,8]#[6,6,6,6,6,6,6,6]#test[2,2,2,2,2,2,2,2] 

confs["vae-bi-1024-attn-addmem-poem"].use_data_queue       = False
confs["vae-bi-1024-attn-addmem-poem"].stddev               = 1.0 
confs["vae-bi-1024-attn-addmem-poem"].addmem               = True
confs["vae-bi-1024-attn-addmem-poem"].bidirectional        = True

confs["cvae-bi-simpleprior-attn-poem"] = copy.deepcopy(confs["vae"])
confs["cvae-bi-simpleprior-attn-poem"].model_kind                    = "CVAERNN"
confs["cvae-bi-simpleprior-attn-poem"].input_vocab_size              = 12000 
confs["cvae-bi-simpleprior-attn-poem"].output_vocab_size             = 12000 

confs["cvae-bi-simpleprior-attn-poem"].learning_rate                 = 0.0001
confs["cvae-bi-simpleprior-attn-poem"].num_layers                    = 3 

confs["cvae-bi-simpleprior-attn-poem"].embedding_size                = 150 
confs["cvae-bi-simpleprior-attn-poem"].cell_model                    = "GRUCell"
confs["cvae-bi-simpleprior-attn-poem"].num_units                     = 1024 
#confs["cvae-bi-simpleprior-attn-poem"].addmem                        = True 
confs["cvae-bi-simpleprior-attn-poem"].bidirectional                 = True 
confs["cvae-bi-simpleprior-attn-poem"].data_dir                      = "/search/odin/Nick/GenerateWorkshop/data/poem"
confs["cvae-bi-simpleprior-attn-poem"].beam_splits                   = [20,20,20,20,20,20,20,20]#[12,12,12,12,12,12,12]#[5,5,5,1,1,1,1]#[3,3,3,1,1,1] 
confs["cvae-bi-simpleprior-attn-poem"].use_data_queue                = False 
confs["cvae-bi-simpleprior-attn-poem"].input_max_len                 = 10
confs["cvae-bi-simpleprior-attn-poem"].output_max_len                = 10 
confs["cvae-bi-simpleprior-attn-poem"].prior_type                    = "simple"
confs["cvae-bi-simpleprior-attn-poem"].enc_latent_dim                = 256 
confs["cvae-bi-simpleprior-attn-poem"].stddev                        = 3.0
confs["cvae-bi-simpleprior-attn-poem"].kld_ratio                     = 1.0
confs["cvae-bi-simpleprior-attn-poem"].bow_ratio                     = None 
