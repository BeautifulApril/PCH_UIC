from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

# from graphviz import Digraph
from torch.autograd import Variable
# from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import itertools
import time
import os
from six.moves import cPickle
from MyDataLoader import *
import opts
import models
from models.networks import *
from models.AttModel import *

from dataloader import *
import eval_utils
import eval_utils_sample
import misc.utils as utils
from misc.obj_rewards import init_scorer, get_self_critical_reward
from models.SentenceEncoder_youhua import *
from models.Decoder import *
import torch.nn.functional

import ipdb
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)
def calculate_simility(a,b):
    scores=F.normalize(a,dim=-1).mm(F.normalize(b,dim=-1).t())
    return scores
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

    labels_all=np.load('./data/captions_train_BOS_EOS.npy')
    np.random.shuffle(labels_all)
    np_start=0
    fake_sen=[]
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    epoch_iter=0
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    netD=SentenceEncoder(opt).cuda()
    dec=models.setup(opt).cuda()

    if vars(opt).get('start_from', None) is not None:
        pth=torch.load(opt.start_from)
        dec.load_state_dict(pth)

    if vars(opt).get('netD_start_from', None) is not None:
        pth=torch.load(opt.netD_start_from)
        netD.load_state_dict(pth['discriminator'])
       

    dec.train()
    netD.train()

    # Assure in training mode
    ###trian GAN####
    ####train or not#######################
    update_lr_flag = True

    ############
    triplet_loss = nn.TripletMarginLoss(margin=opt.margin, p=2,reduction='none')
    rl_crit = utils.RewardCriterion()
    crit = utils.LanguageModelCriterion().cuda()
    criterionGAN = GANLoss(opt.gan_mode).cuda()
    criterionCycle = torch.nn.L1Loss().cuda()
    criterionIdt = torch.nn.L1Loss().cuda()
    optimizer_G = torch.optim.Adam(dec.parameters(), lr=opt.gen_lr, betas=(opt.beta1, 0.999), eps=opt.optim_epsilon, weight_decay=opt.weight_decay)
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.dis_lr, betas=(opt.beta1, 0.999), eps=opt.optim_epsilon, weight_decay=opt.weight_decay)
    det_obj_36max_all_yuzhi_score=np.load('discriminator/det_obj_36max_all_0.1_scores.npy')
    det_obj_36max_all_yuzhi_mask=np.load('discriminator/det_obj_36max_all_0.1_mask.npy')
    det_obj_36max_all_yuzhi_score=det_obj_36max_all_yuzhi_score*det_obj_36max_all_yuzhi_mask
    img_info = json.load(open(opt.input_json))
    split2id=json.load(open('triplet_RL/split2id.json'))
    id2split=json.load(open('triplet_RL/id2split.json'))
    transfer_obj=json.load(open('triplet_RL/data/transfer_obj.json','r'))
    GT_img_attr=np.load('discriminator/det_obj_36max_all_0.1_mask.npy')

    while True:
        if update_lr_flag:
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr_G = opt.gen_lr * decay_factor
                opt.current_lr_D = opt.dis_lr * decay_factor
            else:
                opt.current_lr_G = opt.gen_lr
                opt.current_lr_D = opt.dis_lr
            utils.set_lr(optimizer_G, opt.current_lr_G)
            utils.set_lr(optimizer_D, opt.current_lr_D)
            update_lr_flag = False

        time_start = time.time()
        data = loader.get_batch('train')
        torch.cuda.synchronize() #the correct way to complete time
        tmp = [data['fc_feats'], data['att_feats'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, masks, att_masks = tmp

        query_overlap=np.zeros([opt.batch_size,1600],dtype=np.float32)

        query_sen,_,query_outputs_query,query_1600=dec(fc_feats, att_feats, att_masks, mode='sample')


        GT_gen_sen_obj_id=torch.zeros([query_sen.size(0),1600])

        for i in range(opt.batch_size):
            ix=data['split_ix'][i]
            query_overlap[i]=GT_img_attr[ix]
            for  gj in range(query_sen.shape[1]):
                if str(query_sen[i][gj].item()) in transfer_obj.keys():
                    GT_gen_sen_obj_id[i][transfer_obj[str(query_sen[i][gj].item())]]=1
       
        GT_gen_sen_obj_attr_norepeat=GT_gen_sen_obj_id.cuda()
        query_overlap=torch.from_numpy(query_overlap).cuda()
        
        #negative 
        overlap_mask=torch.mm(query_overlap,query_overlap.t())
        for i in range(overlap_mask.size(0)):
            overlap_mask[i][i] = 0
        similar_mat_id=torch.argmax(overlap_mask,dim=1)
        similar_mat=torch.zeros(opt.batch_size,opt.batch_size).type(torch.LongTensor).cuda()
        for i in range(opt.batch_size):
            similar_mat[i][similar_mat_id[i]]=1

        dis_similar_mat=overlap_mask.eq(0)
        simlarity_mask=calculate_simility(fc_feats,fc_feats)<0.5
        dis_similar_mat=dis_similar_mat*simlarity_mask
        dis_similar_mat.type(torch.LongTensor).cuda()

        query_mask=torch.sum(GT_gen_sen_obj_attr_norepeat.mul(query_overlap),dim=1)>1
        query_mask_mean=torch.mean(query_mask.float()).item()

        if np_start+opt.batch_size >= labels_all.shape[0]:
            np_start=0
            np.random.shuffle(labels_all)
        labels=labels_all[np_start:np_start+opt.batch_size]
        np_start+=opt.batch_size
        labels=torch.from_numpy(labels).cuda()
        # generate mask
        masks = np.zeros([opt.batch_size, opt.seq_length + 2], dtype = 'float32')
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, labels)))
        for ix, row in enumerate(masks):
            row[:nonzeros[ix]] = 1
        masks=torch.from_numpy(masks).cuda()

        print('----------------')

        gen_result, sample_logprobs, _ , _= dec(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')#sample

        dec.eval()  
        with torch.no_grad():    
            fake_sen_output=dec(fc_feats, att_feats, att_masks,opt={'decoding_constraint':1}, mode='sample')[0]#greedy
        dec.train()
        greedy_sample_masks=(torch.sum(torch.abs(gen_result-fake_sen_output),1)>0).reshape(-1,1).float()
 
        # positive_sen,_,positive_outputs,positive_1600=dec(top1_fc_feats, top1_att_feats, att_masks, mode='sample')

        fake_masks=(fake_sen_output>0).float()#greedy_masks
        fake_masks=torch.cat([fake_masks.new(fake_masks.size(0),1).fill_(1),fake_masks[:,:-1]],1)
        sample_masks=(gen_result>0).float()
        sample_masks=torch.cat([sample_masks.new(sample_masks.size(0),1).fill_(1),sample_masks[:,:-1]],1)
        eps=1e-7
        ###clear repeat
        obj_attr_reward_sample=torch.zeros([opt.batch_size,gen_result.size(1)]).cuda()
        obj_attr_reward_sample_no_penalty=torch.zeros([opt.batch_size,gen_result.size(1)]).cuda()
        obj_attr_reward_greedy=torch.zeros([opt.batch_size,fake_sen_output.size(1)]).cuda()
        greedy_len_masks=torch.sum(fake_sen_output>0,1)#greedy_masks
        sample_len_masks=torch.sum(gen_result>0,1)

        for i in range(opt.batch_size):
            numdict={}
            idx=data['split_ix'][i]
            for w in range(sample_len_masks[i].item()):
                if str(gen_result[i][w].item()) in transfer_obj.keys():
                    if gen_result[i][w].item() not in numdict:   
                        obj_attr_reward_sample [i][w]=float(det_obj_36max_all_yuzhi_score[idx][transfer_obj[str(gen_result[i][w].item())]])
                        numdict[gen_result[i][w].item()]=1
                    else:
                        numdict[gen_result[i][w].item()]+=1
                        obj_attr_reward_sample [i][w]=-float(np.abs(det_obj_36max_all_yuzhi_score[idx][transfer_obj[str(gen_result[i][w].item())]]))

            numdict_G={}
            for w in range(greedy_len_masks[i].item()):
                if str(fake_sen_output[i][w].item()) in transfer_obj.keys():
                    if fake_sen_output[i][w].item() not in numdict_G:        
                        obj_attr_reward_greedy [i][w]=float(det_obj_36max_all_yuzhi_score[idx][transfer_obj[str(fake_sen_output[i][w].item())]])
                        numdict_G[fake_sen_output[i][w].item()]=1
                    else:
                        numdict_G[fake_sen_output[i][w].item()]+=1
                        obj_attr_reward_greedy [i][w]=-float(np.abs(det_obj_36max_all_yuzhi_score[idx][transfer_obj[str(fake_sen_output[i][w].item())]]))
        #backward_D_A()calculate gradients for D_A
        for i in range(opt.range_netD):
            set_requires_grad(netD, True)
            optimizer_D.zero_grad()
            hidden_sentence,real_probs=netD(labels[:,1:])
            fake_sen_output_EOS = torch.cat([fake_sen_output.detach(),fake_sen_output.detach().new(fake_sen_output.detach().size(0),1).fill_(0)],1)
            pre_fake_sen,fake_probs=netD(fake_sen_output_EOS.detach())
            gen_result_EOS = torch.cat([gen_result.detach(),gen_result.detach().new(gen_result.detach().size(0),1).fill_(0)],1)
            pre_gen_sen,sample_fake_probs=netD(gen_result_EOS)

            real_len=torch.sum(labels[:,1:]!=0,1)
            real_target = torch.zeros(real_probs.size()[0], real_probs.size()[1]).scatter_(1, (real_len).cpu().unsqueeze(-1), 1).cuda()
            real_probs_1d=torch.sum(real_probs*real_target,1).reshape(opt.batch_size,-1)

            fake_len=torch.sum(fake_sen_output.detach()!=0,1)
            fake_target=torch.zeros(fake_probs.size()[0], fake_probs.size()[1]).scatter_(1, (fake_len).cpu().unsqueeze(-1), 1).cuda()
            fake_probs_1d=torch.sum(fake_probs*fake_target,1).reshape(opt.batch_size,-1)

            sample_len=torch.sum(gen_result.detach()!=0,1)
            sample_target=torch.zeros(sample_fake_probs.size()[0], sample_fake_probs.size()[1]).scatter_(1, (sample_len).cpu().unsqueeze(-1), 1).cuda()
            D_sample_probs_1d=torch.sum(sample_fake_probs*sample_target,1).reshape(opt.batch_size,-1)

            loss_f=nn.BCELoss(reduce=False, size_average=True).cuda()
            smooth=0.25
            loss_D_real = loss_f(real_probs_1d, torch.ones_like(real_probs_1d)*(1-smooth) + 0.5 * smooth).mean()
            loss_D_fake=loss_f(fake_probs_1d, torch.zeros_like(fake_probs_1d)).mean()
            loss_D_sample= torch.sum(loss_f(D_sample_probs_1d, torch.zeros_like(D_sample_probs_1d))*greedy_sample_masks)/(torch.sum(greedy_sample_masks)+eps)
            loss_D=0.5*loss_D_real+0.5*loss_D_sample*(torch.sum(greedy_sample_masks)/opt.batch_size)

            loss_D.backward()
            optimizer_D.step()
        ###end backward netD###
        set_requires_grad(netD, False)
        optimizer_G.zero_grad()
        fake_sen_output=torch.cat([fake_sen_output,fake_sen_output.new(fake_sen_output.size(0),1).fill_(0)],1)
        pre_fake_sen,greedy_fake_probs=netD(fake_sen_output)
        gen_result = torch.cat([gen_result,gen_result.new(gen_result.size(0),1).fill_(0)],1)
        pre_gen_sen,sample_fake_probs=netD(gen_result)

        fake_len=torch.sum(fake_sen_output.detach()!=0,1)
        fake_target=torch.zeros(greedy_fake_probs.size()[0], greedy_fake_probs.size()[1]).scatter_(1, (fake_len).cpu().unsqueeze(-1), 1).cuda()
        greedy_probs_1d=torch.sum(greedy_fake_probs*fake_target,1).reshape(opt.batch_size,-1)

        sample_len=torch.sum(gen_result.detach()!=0,1)
        sample_target=torch.zeros(sample_fake_probs.size()[0], sample_fake_probs.size()[1]).scatter_(1, (sample_len).cpu().unsqueeze(-1), 1).cuda()
        sample_probs_1d=torch.sum(sample_fake_probs*sample_target,1).reshape(opt.batch_size,-1)  

        # backward_G()

        loss_reconstruct=0
        reward_sample=torch.log(sample_probs_1d+eps)
        reward_greedy=torch.log(greedy_probs_1d+eps)
        reward_G=reward_sample-reward_greedy
        reward_G=reward_G.cpu().detach().numpy()
        reward_G=np.repeat(reward_G[:,np.newaxis],gen_result.size(1)-1,1)
        # ipdb.set_trace()
        loss_G = rl_crit(sample_logprobs, gen_result[:,:-1].data, torch.from_numpy(reward_G).float().cuda())
        
        obj_attr_reward_sample=torch.sum(obj_attr_reward_sample*sample_masks,1)
        obj_attr_reward_greedy=torch.sum(obj_attr_reward_greedy*fake_masks,1)
        reward_object=obj_attr_reward_sample-obj_attr_reward_greedy
        reward_object=reward_object.cpu().detach().numpy()
        reward_object=np.repeat(reward_object[:,np.newaxis],gen_result[:,:-1].size(1),1)
        loss_object = rl_crit(sample_logprobs, gen_result[:,:-1].data, torch.from_numpy(reward_object).float().cuda())
        
        query=query_1600

        video_cos = calculate_simility(query,query)
        video_l2_distance = 2.0 - 2.0 * video_cos
        video_l2_distance = video_l2_distance

        trip_similar_mat = similar_mat.view(opt.batch_size,
                                            opt.batch_size, 1)
        trip_similar_mat = trip_similar_mat.expand(opt.batch_size,
                                                   opt.batch_size,
                                                   opt.batch_size)
        trip_dis_similar_mat = dis_similar_mat.view(opt.batch_size, 1,
                                                    opt.batch_size)
        trip_dis_similar_mat = trip_dis_similar_mat.expand(opt.batch_size,
                                                           opt.batch_size,
                                                           opt.batch_size)
        # trip_dis_similar_mat = trip_dis_similar_mat.view(Param.batch_size*Param.triplet_num_per_class*Param.batch_size*Param.triplet_num_per_class, Param.batch_size*Param.triplet_num_per_class)
        # ipdb.set_trace()
        trip_mat = trip_similar_mat.long()* trip_dis_similar_mat.long()
        trip_mat = trip_mat.type(torch.FloatTensor)
        tot_num = trip_mat.sum()
        similar_mat = similar_mat.type(torch.FloatTensor)
        tot_ij_num = similar_mat.sum()

        trip_mat = Variable(trip_mat.cuda())

        ##########  cuda  ##########

        trip_s = video_l2_distance.view(opt.batch_size,
                                        opt.batch_size, 1)
        trip_s = trip_s.expand(opt.batch_size,
                               opt.batch_size,
                               opt.batch_size)
        trip_s = trip_s.contiguous()
        trip_d = video_l2_distance.view(opt.batch_size, 1,
                                        opt.batch_size)
        trip_d = trip_d.expand(opt.batch_size,
                               opt.batch_size,
                               opt.batch_size)
        trip_d = trip_d.contiguous()
        # trip_d = trip_d.view(Param.batch_size*Param.triplet_num_per_class*Param.batch_size*Param.triplet_num_per_class, Param.batch_size*Param.triplet_num_per_class)
        
        trip_sd_margin_relu = F.relu(trip_s - trip_d + 0.2)
        triplet_loss = trip_sd_margin_relu * trip_mat

        trip_sd_margin = (trip_s - trip_d + 0.2) * opt.triplet_beta
        exp_trip_sd_margin = torch.exp( trip_sd_margin )
        t_exp_trip_sd_margin = exp_trip_sd_margin.permute(1,0,2)
        exp_trip = exp_trip_sd_margin + t_exp_trip_sd_margin
        exp_trip = exp_trip * trip_mat
        exp_trip_reduce = exp_trip.sum(dim = 2)
        log_exp_trip_loss = torch.log1p(exp_trip_reduce).sum()
        log_exp_trip_loss /= tot_ij_num         
        # ipdb.set_trace()       
        # concept_per_sample=torch.mean(torch.sum(obj_attr_reward_sample_no_penalty,dim=1)).item()
        # concept_per_greedy=torch.mean(torch.sum(obj_attr_reward_sample_no_penalty,dim=1)).item()
        
        loss_G_all = loss_G*opt.lambda_G+loss_object*opt.lambda_obj+log_exp_trip_loss*opt.lambda_tpt
        
        loss_G_all.backward()
        optimizer_G.step()



        print('epoch:',epoch,'iteration',iteration,'np_start',np_start,'loss_G_all',loss_G_all,'loss_G',loss_G,'loss_object',loss_object,)
        print('reward_G:', np.mean(reward_G),'reward_sample',torch.mean(reward_sample),'reward_greedy',torch.mean(reward_greedy))
        print('reward_object',np.mean(reward_object),'obj_attr_reward_sample',torch.mean(obj_attr_reward_sample),'obj_attr_reward_greedy',torch.mean(obj_attr_reward_greedy))
        print('--loss_D',loss_D,'loss_D_real',loss_D_real,'loss_D_fake',loss_D_fake)
        print('real_probs', torch.mean(real_probs_1d),'fake_probs', torch.mean(fake_probs_1d),'D_sample_probs_1d', torch.mean(D_sample_probs_1d))
        print('loss_triplet', log_exp_trip_loss)

        iteration += 1
        epoch_iter += 1
        if data['bounds']['wrapped']:     
            update_lr_flag = True
            epoch_iter=0  
            epoch += 1

        # Write the training loss summary

        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'loss_G', loss_G, iteration)
            add_summary_value(tb_summary_writer, 'loss_object', loss_object, iteration)
            add_summary_value(tb_summary_writer, 'loss_G_all', loss_G_all, iteration)
            add_summary_value(tb_summary_writer, 'loss_D_real', loss_D_real, iteration)
            add_summary_value(tb_summary_writer, 'loss_D_fake', loss_D_fake, iteration)
            add_summary_value(tb_summary_writer, 'loss_D_sample', loss_D_sample, iteration)
            add_summary_value(tb_summary_writer, 'loss_D', loss_D, iteration)
            add_summary_value(tb_summary_writer, 'reward_G:', np.mean(reward_G), iteration)
            add_summary_value(tb_summary_writer, 'reward_object:', np.mean(reward_object), iteration)
            add_summary_value(tb_summary_writer, 'reward_sample:', torch.mean(reward_sample),iteration)
            add_summary_value(tb_summary_writer, 'reward_greedy:', torch.mean(reward_greedy), iteration)
            add_summary_value(tb_summary_writer, 'obj_attr_reward_sample', torch.mean(obj_attr_reward_sample), iteration)
            add_summary_value(tb_summary_writer, 'obj_attr_reward_greedy', torch.mean(obj_attr_reward_greedy), iteration)
            add_summary_value(tb_summary_writer, 'real_probs', torch.mean(real_probs_1d), iteration)
            add_summary_value(tb_summary_writer, 'fake_probs', torch.mean(fake_probs_1d), iteration)
            add_summary_value(tb_summary_writer, 'D_sample_probs_1d', torch.mean(D_sample_probs_1d), iteration)
            add_summary_value(tb_summary_writer, 'greedy_probs_1d', torch.mean(greedy_probs_1d), iteration)
            add_summary_value(tb_summary_writer, 'sample_probs_1d', torch.mean(sample_probs_1d), iteration)
            add_summary_value(tb_summary_writer, 'loss_triplet', log_exp_trip_loss, iteration)
            add_summary_value(tb_summary_writer, 'query_mask_mean', query_mask_mean, iteration)
        if (iteration % opt.save_checkpoint_every == 0):
            eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json,
                            }               
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dec, crit, loader, eval_kwargs)
            # add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)

            # eval_kwargs_sample = {'split': 'test',
            #                 'dataset': opt.input_json,
            #                 'sample_max':0
            #                 } 
            # eval_kwargs_sample.update(vars(opt))
            # val_loss_sample, predictions_sample, lang_stats_sample = eval_utils.eval_split(dec, crit, loader, eval_kwargs_sample)
            # if lang_stats_sample is not None:
            #     for k,v in lang_stats_sample.items():
            #         add_summary_value(tb_summary_writer, k+'_sample', v, iteration)

            state={
                'discriminator':netD.state_dict(),
                'generator':dec.state_dict(),
                'epoch':epoch
                 }
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model'+str(epoch)+'.pth')
            # checkpoint_path = os.path.join(opt.checkpoint_path, 'model_'+str(iteration)+'.pth')
            torch.save(state, checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            best_flag = False
            current_score=lang_stats['CIDEr']
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(state, checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
            # if not os.path.isdir(os.path.join('fake_sen',opt.id)):
            #    os.mkdir(os.path.join('fake_sen',opt.id)) 
            # np.save(os.path.join('fake_sen/'+str(opt.id),'fake_sen'+str(iteration)),fake_sen)

        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break    

    
opt = opts.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES']=str(opt.GPU)
train(opt)
