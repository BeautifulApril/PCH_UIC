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
To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).
### Evaluate
```
$ sh eval/test.sh 
```
The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_method greedy`), to sample from the posterior, set `--sample_method sample`.

