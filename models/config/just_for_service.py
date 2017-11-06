from Config import Config
from Config import confs
# here is the config of models exported from other places(not trained here), for the use of servers

# Tsinghua fake config
confs["tsinghua"] = Config()
confs["tsinghua"].model_kind                        = "Tsinghua"
confs["tsinghua"].input_max_len                     = 40
confs["tsinghua"].output_max_len                    = 12 
confs["tsinghua"].beam_size                         = 5 

confs["postprob"] = Config() 
confs["postprob"].model_kind                        = "Postprob"
confs["postprob"].input_max_len                     = 40 
confs["postprob"].output_max_len                    = 30 

# tianchuan fake config
confs["tianchuan_cnn"] = Config()
confs["tianchuan_cnn"].model_kind = "KimCNN"

confs["tianchuan_cnn_tag"] = Config()
confs["tianchuan_cnn_tag"].model_kind = "KimCNN" 
