from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
import os
matplotlib.use('Agg')
import sys
sys.path .append(os.path.abspath('.'))


from torch.autograd import Variable
# from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import itertools
import time

from six.moves import cPickle
from MyDataLoader import *
import opts
import models
from models.networks import *
from models.AttModel import *
# import models.networks
from dataloader import *
import eval_utils
# import pseudo_eval_utils
import misc.utils as utils
from misc.obj_rewards import init_scorer, get_self_critical_reward
from models.SentenceEncoder import *
from models.Decoder import *
import torch.nn.functional
# import object_score
import ipdb
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    loader = MyDataLoader(opt)
    eval_loader=DataLoader(opt)

    json_info = json.load(open(opt.input_json))
    ix_to_word = json_info['ix_to_word']
    opt.vocab_size = len(ix_to_word)
    opt.seq_length = 16

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)


    dec=models.setup(opt).cuda()

    if vars(opt).get('start_from', None) is not None:
        pth=torch.load(opt.start_from)
        dec.load_state_dict(pth['generator'])
        # dec.load_state_dict(pth)

    dec.eval()
    dec.cuda()
    crit = utils.LanguageModelCriterion()

    eval_kwargs = {'split': 'test',
                    'dataset': opt.input_json}               
    eval_kwargs.update(vars(opt))
    # ipdb.set_trace()
    val_loss, predictions, lang_stats = eval_utils.eval_split(dec, crit, loader, eval_kwargs)
    ipdb.set_trace()
    
opt = opts.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES']=str(opt.GPU)
train(opt)
