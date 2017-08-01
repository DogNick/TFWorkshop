import copy
from Config import confs
from Config import Config 

confs["stc-2-contrastive"] = Config()
confs["stc-2-contrastive"].model_kind = "DeepMatchContrastive"
confs["stc-2-contrastive"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-multi-neg"
confs["stc-2-contrastive"].embedding_init = "/search/odin/Nick/GenerateWorkshop/data/STC-2/Nick_stc_pair.w2v.npy"
confs["stc-2-contrastive"].learning_rate       = 0.5 
confs["stc-2-contrastive"].input_vocab_size    = 115205 
confs["stc-2-contrastive"].batch_size          = 128
confs["stc-2-contrastive"].opt_name            = "Adadelta" 
#confs["stc-2-contrastive"].opt_name            = "SGD" 
confs["stc-2-contrastive"].max_to_keep         = 10
confs["stc-2-contrastive"].input_max_len       = 35
confs["stc-2-contrastive"].num_layers          = 1 
confs["stc-2-contrastive"].num_units           = 128 
confs["stc-2-contrastive"].embedding_size      = 150
confs["stc-2-contrastive"].cell_model          = "LSTMCell"
confs["stc-2-contrastive"].m1                  = 3 
confs["stc-2-contrastive"].c1                  = 256 
confs["stc-2-contrastive"].replicas_to_aggregate = 7 
confs["stc-2-contrastive"].total_num_replicas = 7 
confs["stc-2-contrastive"].cluster = {
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
            "0.0.0.0:4006",
            #"0.0.0.0:4007",
        ] 
}
confs["stc-2-contrastive-strong-init"] = confs["stc-2-contrastive"]
#########################################################
confs["stc-2-interact"] = Config()
confs["stc-2-interact"].max_to_keep         = 10
confs["stc-2-interact"].model_kind = "DeepMatchInteract"
confs["stc-2-interact"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-multi-neg"
confs["stc-2-interact"].embedding_init = "/search/odin/Nick/GenerateWorkshop/data/STC-2/Nick_stc_pair.w2v.npy"
confs["stc-2-interact"].learning_rate       = 0.5 
confs["stc-2-interact"].input_vocab_size    = 115205 
confs["stc-2-interact"].batch_size          = 128
confs["stc-2-interact"].opt_name            = "Adadelta" 
confs["stc-2-interact"].input_max_len       = 35
confs["stc-2-interact"].num_layers          = 1 
confs["stc-2-interact"].num_units           = 128 
confs["stc-2-interact"].embedding_size      = 150
confs["stc-2-interact"].cell_model          = "LSTMCell"
confs["stc-2-interact"].output_keep_prob    = 0.8 
confs["stc-2-interact"].conv_conf = [
                        [(3, 3, 1, 128), (1, 1), (2, 2), (2, 2)],
                        [(2, 2, 128, 64), (1, 1), (2, 2), (2, 2)]
                      ]
confs["stc-2-interact"].fc_h_size           = 100 
confs["stc-2-interact"].replicas_to_aggregate = 7 
confs["stc-2-interact"].total_num_replicas = 7 
confs["stc-2-interact"].cluster = {
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
            "0.0.0.0:4006",
            #"0.0.0.0:4007",
        ] 
}
################################################################
confs["stc-2-interact-qppr"] = copy.deepcopy(confs["stc-2-interact"])
confs["stc-2-interact-qppr"].num_layers = 2
confs["stc-2-interact-qppr"].model_kind = "DeepMatchInteractQPPR"
confs["stc-2-interact-qppr"].num_units = 512
confs["stc-2-interact-qppr"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-multi-neg"
################################################################
confs["stc-2-interact-qppr-2"] = copy.deepcopy(confs["stc-2-interact"])
confs["stc-2-interact-qppr-2"].num_layers = 2
confs["stc-2-interact-qppr-2"].model_kind = "DeepMatchInteractQPPR"
confs["stc-2-interact-qppr-2"].num_units = 512
confs["stc-2-interact-qppr-2"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-q_pq_r_negr"
###############################################################
confs["stc-2-interact-qpr-negpr"] = copy.deepcopy(confs["stc-2-interact"])
confs["stc-2-interact-qpr-negpr"].num_layers = 2
confs["stc-2-interact-qpr-negpr"].model_kind = "DeepMatchInteractQPRnegPR"
confs["stc-2-interact-qpr-negpr"].num_units = 1024
confs["stc-2-interact-qpr-negpr"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-q_p_r_negpr"
###############################################################
confs["stc-2-interact-qpr-negpr-shuf"] = copy.deepcopy(confs["stc-2-interact"])
confs["stc-2-interact-qpr-negpr-shuf"].model_kind = "DeepMatchInteractQPRnegPR"
confs["stc-2-interact-qpr-negpr-shuf"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-q-pr-negpr-shuf"
confs["stc-2-interact-qpr-negpr-shuf"].num_layers = 3
confs["stc-2-interact-qpr-negpr-shuf"].num_units = 1024
###############################################################
confs["stc-2-interact-qpr-negpr-obj0"] = copy.deepcopy(confs["stc-2-interact"])
confs["stc-2-interact-qpr-negpr-obj0"].num_layers = 2
confs["stc-2-interact-qpr-negpr-obj0"].model_kind = "DeepMatchInteractQPRnegPR"
confs["stc-2-interact-qpr-negpr-obj0"].num_units = 1024
confs["stc-2-interact-qpr-negpr-obj0"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-q_p_r_negpr"
#############################################################
confs["stc-2-interact-concat-rnn"] = Config()
confs["stc-2-interact-concat-rnn"].max_to_keep         = 10
confs["stc-2-interact-concat-rnn"].model_kind = "DeepMatchInteractConcatRNN"
confs["stc-2-interact-concat-rnn"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2-10neg"
confs["stc-2-interact-concat-rnn"].embedding_init = "/search/odin/Nick/GenerateWorkshop/data/STC-2/Nick_stc_pair.w2v.npy"
confs["stc-2-interact-concat-rnn"].learning_rate       = 0.5 
confs["stc-2-interact-concat-rnn"].input_vocab_size    = 115205 
confs["stc-2-interact-concat-rnn"].batch_size          = 128
confs["stc-2-interact-concat-rnn"].opt_name            = "Adadelta" 
confs["stc-2-interact-concat-rnn"].input_max_len       = 35
confs["stc-2-interact-concat-rnn"].num_layers          = 1 
confs["stc-2-interact-concat-rnn"].num_units           = 128 
confs["stc-2-interact-concat-rnn"].embedding_size      = 150
confs["stc-2-interact-concat-rnn"].cell_model          = "LSTMCell"
confs["stc-2-interact-concat-rnn"].output_keep_prob    = 0.8 
confs["stc-2-interact-concat-rnn"].conv_conf = [
                        [(3, 3, 1, 128), (1, 1), (2, 2), (2, 2)],
                        [(2, 2, 128, 64), (1, 1), (2, 2), (2, 2)]
                     ]
confs["stc-2-interact-concat-rnn"].fc_h_size           = 100
#########################################################
confs["stc-2"] = Config()
confs["stc-2"].model_kind = "DeepMatch"
confs["stc-2"].data_dir = "/search/odin/Nick/GenerateWorkshop/data/STC-2"
confs["stc-2"].embedding_init = "/search/odin/Nick/GenerateWorkshop/data/STC-2/Nick_stc_for_w2v.w2v.npy"
confs["stc-2"].learning_rate       = 2.0 
confs["stc-2"].input_vocab_size    = 100000 
confs["stc-2"].batch_size          = 128
#confs["stc-2"].opt_name            = "RMSProp" 
confs["stc-2"].opt_name            = "Adadelta" 
#confs["stc-2"].opt_name            = "SGD" 
confs["stc-2"].max_to_keep         = 10
confs["stc-2"].input_max_len       = 30 
confs["stc-2"].num_layers          = 1 
confs["stc-2"].num_units           = 128 
confs["stc-2"].embedding_size      = 200 
confs["stc-2"].cell_model          = "GRUCell"
confs["stc-2"].m1                  = 3 
confs["stc-2"].c1                  = 32 
confs["stc-2"].replicas_to_aggregate = 7 
confs["stc-2"].total_num_replicas = 7 
confs["stc-2"].cluster = {
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
            "0.0.0.0:4006",
            #"0.0.0.0:4007",
        ] 
}
