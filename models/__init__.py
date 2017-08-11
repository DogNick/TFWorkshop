import logging as log
import codecs
import re
import os
import numpy as np

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import flags
from tensorflow.python.ops import variable_scope

#from DeepMatch import * 
#from DeepMatchContrastive import *
#from DeepMatchInteract import *
#from DeepMatchInteractConcatRNN import *
#from DeepMatchInteractQPPR import *
#from DeepMatchInteractQPRnegPR import *

#from DynAttnTopicSeq2Seq import *
from AttnSeq2Seq import *
from VAERNN import * 
from VAERNN2 import * 
from CVAERNN import *
from AllAttn import *
from RNNClassification import *

from Tsinghua_plan import *
from Nick_plan import *

from QueueReader import *
from config import *
from util import *

FLAGS = tf.app.flags.FLAGS

# get graph log
graphlg = log.getLogger("graph")

magic = {
        #"DeepMatch": DeepMatch,
        #"DeepMatchContrastive": DeepMatchContrastive,
        #"DeepMatchInteract": DeepMatchInteract,
        #"DeepMatchInteractQPPR": DeepMatchInteractQPPR,
        #"DeepMatchInteractQPRnegPR": DeepMatchInteractQPRnegPR,
        #"DeepMatchInteractConcatRNN": DeepMatchInteractConcatRNN,

        #"DynAttnTopicSeq2Seq": DynAttnTopicSeq2Seq,
        "AttnSeq2Seq": AttnSeq2Seq,
        "VAERNN": VAERNN,
        "VAERNN2": VAERNN2,
        "CVAERNN": CVAERNN,
        "RNNClassification": RNNClassification,
        "Postprob": Postprob,
        "Tsinghua": Tsinghua,
        "AllAttn": AllAttn 
}

def create(conf_name, job_type="single", task_id="0", dtype=tf.float32):
    return magic[confs[conf_name].model_kind](conf_name, job_type, task_id, dtype)

