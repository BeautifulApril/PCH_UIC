# Prepare data

Note: every preprocessed file or preextracted features can be found in [link](https://drive.google.com/open?id=1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J).

## COCO

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.


### Image features option 1: Resnet features

#### Download COCO dataset and pre-extract the image features(if you want to extract your self)

Download pretrained resnet models. The models can be downloaded from [here](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and should be placed in `data/imagenet_weights`.

Download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the `train2014/` and `val2014/` in the same directory, denoted as `$IMAGE_ROOT`.



(Check the prepro scripts for more options, like other resnet models or other attention sizes.)

**Warning**: the prepro script will fail with the default MSCOCO data because one of their images is corrupted. See [this issue](https://github.com/karpathy/neuraltalk2/issues/4) for the fix, it involves manually replacing one image in the dataset.

#### Download preextracted features

To skip the preprocessing, you can download and decompress `cocotalk_att.tar` and `cocotalk_fc.tar` from the link provided at the beginning.)

### Image features option 2: Bottom-up features (current standard)

#### Convert from peteanderson80's original file
Download pre-extracted features from [link](https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one or fixed one.


This will create `data/cocobu_fc`(not necessary), `data/cocobu_att` and `data/cocobu_box`. If you want to use bottom-up feature, you can just replace all `"cocotalk"` with `"cocobu"` in the training/test scripts.

#### Download converted files

bottomup-att: [link](https://drive.google.com/file/d/1hun0tsel34aXO4CYyTRIvHJkcbZHwjrD/view?usp=sharing)

### Image features option 3:  Vilbert 12 in 1 features.
In vilbert-12-in-1, the image features used is similar to the original bottom-up feature but with a model with renext152 backbone.

Here is the link of the converted lmdb(More compressed than the original one provided by jiasen):

[https://drive.google.com/file/d/1Gjo9Xs7qrjah2TQs0-joEWi8HabCkuQp/view?usp=sharing](https://drive.google.com/file/d/1Gjo9Xs7qrjah2TQs0-joEWi8HabCkuQp/view?usp=sharing)

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