# PCH_UIC
Anonymous code for ICMR2024
## Requirements
- Python 3
- PyTorch 1.3+ (along with torchvision) (Test with 1.13)
- cider
- coco-caption

### Prepare data
We now support both flickr30k and COCO. See details in [data/README.md](data/README.md).

### Start training
```
$ sh train/train.sh 
```

### Evaluate
```
$ sh eval/eval.sh 
```


