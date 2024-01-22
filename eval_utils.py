from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import ipdb
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import pandas as pd
def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    # annFile = 'coco-caption/annotations/captions_val2014f.json'
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results', model_id + '_' + split + '.json')
    cache_path_cider = os.path.join('eval_results', model_id + '_' + split + 'cider.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # ipdb.set_trace()
    imgToEval = cocoEval.imgToEval
    preds_csv=[]
    for p in preds_filt:
        preds_dict={}
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
        preds_dict['image_id']=image_id
        preds_dict['CIDEr']=imgToEval[image_id]['CIDEr']
        preds_dict['caption']=caption
        preds_csv.append(preds_dict)

    df=pd.DataFrame(preds_csv,columns=['image_id','CIDEr','caption'])
    df_cache_path=os.path.join('eval_results',model_id+'.csv')
    df.to_csv(df_cache_path,index=False)
    # ipdb.set_trace()

    with open(cache_path_cider, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)


    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)
    # ipdb.set_trace()
    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    new_pseudo_pair=np.zeros([123287,18],np.int32)
    beam5_results=[]
    beam1_results=[]
    img=json.load(open('data/cocotalk.json'))['images']
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        # ipdb.set_trace()
        # if data.get('labels', None) is not None and verbose_loss:
        #     # forward the model to get loss
        #     tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        #     tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        #     fc_feats, att_feats, labels, masks, att_masks = tmp

        #     with torch.no_grad():
        #         loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
        #     loss_sum = loss_sum + loss
        #     loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        loader.seq_per_img=1
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
        # ipdb.set_trace()    
        # for i in range(seq.size(0)):
        #     new_pseudo_pair[data['split_ix'][i]][1:-1]=seq[i].cpu().numpy()
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                # beam5_dict={}
                # beam5_dict['image_id']=img[data['split_ix'][i]]['id']
                # beam5_dict['captions']=[utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]
                # beam5_results.append(beam5_dict)
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
                
                # ipdb.set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        # for i in range(loader.batch_size):
        #     beam1_dict={}
        #     beam1_dict['image_id']=img[data['split_ix'][i]]['id']
        #     beam1_dict['captions']=[sents[i]]
        #     beam1_results.append(beam1_dict)
        
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            # ipdb.set_trace()
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
