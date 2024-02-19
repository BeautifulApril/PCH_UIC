# Prepare data

## COCO

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.


### Image features option 1: Resnet features

#### Download COCO dataset and pre-extract the image features(if you want to extract your self)

Download pretrained resnet models. 
Download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the `train2014/` and `val2014/` in the same directory, denoted as `$IMAGE_ROOT`.

(Check the prepro scripts for more options, like other resnet models or other attention sizes.)

#### Download preextracted features

To skip the preprocessing, you can download and decompress `cocotalk_att.tar` and `cocotalk_fc.tar` from the link provided at the beginning.)

### Image features option 2: Bottom-up features (current standard)

#### Convert from peteanderson80's original file
Download pre-extracted features from [link](https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one or fixed one.


This will create `data/cocobu_fc`(not necessary), `data/cocobu_att` and `data/cocobu_box`. If you want to use bottom-up feature, you can just replace all `"cocotalk"` with `"cocobu"` in the training/test scripts.

## Flickr30k.

### Feature extraction

For resnet feature, you can do the same thing as COCO.

For bottom-up feature, you can download from [link](https://github.com/kuanghuei/SCAN)

`wget https://scanproject.blob.core.windows.net/scan-data/data.zip`

and then convert to a pth file using the following script:

```
import numpy as np
import os
import torch
from tqdm import tqdm

out = {}
def transform(id_file, feat_file):
  ids = open(id_file, 'r').readlines()
  ids = [_.strip('\n') for _ in ids]
  feats = np.load(feat_file)
  assert feats.shape[0] == len(ids)
  for _id, _feat in tqdm(zip(ids, feats)):
    out[str(_id)] = _feat

transform('dev_ids.txt', 'dev_ims.npy')
transform('train_ids.txt', 'train_ims.npy')
transform('test_ids.txt', 'test_ims.npy')

torch.save(out, 'f30kbu_att.pth')
```
