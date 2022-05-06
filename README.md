# siamrpn.pytorch
This repository is clean re-implement of siamrpn using PyTorch that is more readable for newers.
这个repo就看看就行，主要是为了简便看懂siamrpn设计的，有些人反馈lmbd读取有问题不能训练，之后有同样问题的就不用反馈了
## Environment
- Ubuntu 18.04
- PyTorch1.2
- CUDA10.0 + CUDNN7.4.2 (not so strict)
- TO install needed packages: `pip install -r requirements.txt`
## Prepare dataset
There are two ways to prepare datasets:
1. This is the easier one. Just download my prepared three files(two `.mdb` and one `.json`) 
at [link1](https://cloud.189.cn/t/EFR7JrBbuqEf), [link2](https://cloud.189.cn/t/NvQJ7b6VBnau) and [link3](https://cloud.189.cn/t/7fUraieyiAjm),
and put them all in `dataset` directory
2. This is the another one: refer to pysot dataset [part](https://github.com/STVIR/pysot/blob/master/TRAIN.md#prepare-training-dataset) and prepare
the **YOUTUBEBB** and **VID**. Then, you need modify some path items in `config.config.py` like me:
> __C.DATASET.VID.ROOT = '/home/lz/Videos/VID/crop511'   
> __C.DATASET.VID.ANNO = '/home/lz/PycharmProjects/pysot-master/training_dataset/vid/train.json'  
> __C.DATASET.YOUTUBEBB.ROOT = '/home/lz/Videos/yt_bb/crop511'  
> __C.DATASET.YOUTUBEBB.ANNO = '/home/lz/PycharmProjects/pysot-master/training_dataset/yt_bb/train.json'

and `cd dataset` run `python prepare_dataset.py`, after 1.5 hours you will get ready like first way.
## Train
1. cd project root directory  
2. you can modify some items, such as **BATCH_SIZE**,**NUM_WORKERS** in  `config/config.yaml`  
3. then run 
```bash
python train.py
```
## Test
This repo use got-10k toolkit to evaluate performance in OTB benchmark, so
you need to do follow things:  
1. Download the raw OTB dataset and unzip all videos(otherwise will download and unzip automatically)
2. run 
```bash
python test_OTB.py --root_dir='your_OTB_dir_in_step1'
``` 
## Result
I have just make it work and need more works to train well, so the result will not provide including pretrained model
and OTB100 performance now. See **TODO**
## TODO
- [x] train and test phase
- [ ] hyper-parameters search
- [ ] multi-GPU training
- [ ] higher performance
