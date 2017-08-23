import os
class Config:
    def __init__(self):
        # TRAINING PROCESS
        self.learning_rate = 0.8
        self.learning_rate_decay_factor = 0.99
        self.lr_check_steps = 1000
        self.lr_keep_steps = 1000
        self.opt = "SGD"
        self.max_gradient_norm = 5.0
        self.batch_size = 128
        self.spec_path = ""
        # DATA READING
        self.max_train_data_size = 100000 
        self.data_dir = ""
        self.batch_size = 128
        self.embedding_init = ""
        self.use_data_queue = True 

        # MODEL HYPERPARAMS
        self.model_kind = ""
        self.input_vocab_size = 0
        self.output_vocab_size = 0
        self.num_layers = 0
        self.num_units = 0
        self.embedding_size = 0
        self.cell_model = "LSTMCell"
        self.attention = "Luo"
        self.out_layer_size = None
        self.use_peephole = False
        self.keep_prob = 0.8 
        self.bidirectional = True
        self.num_samples = 512
        self.beam_splits = [1] 
        self.enc_reverse = True
        self.restore_conf = None 
        self.reverse = False
        self.addmem = False
        self.dec_init_type = "each2each" # each2each, all2first, allzeros 
        self.use_init_proj = True # project final enc_states to dec init states 
        self.kld_ratio = 0.000
        self.max_res_num = 70 
        self.visual_data_filename = None
        self.visual_tensor = None
        self.stddev = 1.0
        self.output_max_len = 0 
        self.input_max_len = 0 
        self.sample_prob = 0.0 

        # DISTRIBUTED 
        self.cluster = None
        self.replicas_to_aggregate = 3
        self.total_num_replicas = 1 
        self.tokenize_mode = "char" 
        self.vocab_dir = ""

confs = {}
