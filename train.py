# In this script, train the network, including:
# dataloader, loss.backward, lr_schedule, freeze_backbone
# save model, write relevant scalars

# TODO: re-write dataset __len__, now just use all
import os
import sys
import json
import time
import lmdb
import torch
from tqdm import tqdm
import torch.nn as nn
from loguru import logger
from model import SiamRPN
from config.config import cfg
from datasets import VIDYTBBLMDB
from lr_scheduler import LogScheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from utils.load_state import load_pretrained_and_check
from loss import cls_cross_entropy_loss, reg_smooth_l1_loss



def build_SGD(cfg, model):
    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpnhead.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer

# init something
# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

log_dir = cfg.TRAIN.LOG_DIR
if not os.path.exists(log_dir): os.makedirs(log_dir)
logger.configure(
        handlers=[
            dict(sink=sys.stderr, level="INFO"),
            dict(sink=os.path.join(log_dir, "train_log.txt"),
                 enqueue=True,
                 serialize=True,
                 diagnose=True,
                 backtrace=True,
                 level="INFO")
        ],
        extra={"common_to_all": "default"},
    )

tb_writer = SummaryWriter(log_dir=log_dir)
cfg.merge_from_file('./config/config.yaml')
cfg.freeze()
logger.info("Merge config file from ./config/config.yaml")
logger.info("config \n{}".format(json.dumps(cfg, indent=4)))


# generate model
logger.info('Start building SiamRPN model')
anchor_num = cfg.ANCHOR.ANCHOR_NUM  # 5
siamrpn = SiamRPN(anchor_num).cuda().train()
logger.info('Finish building SiamRPN model')

# construct dataset and dataloader
logger.info('Start building dataset')
db = lmdb.open(path='dataset', readonly=True, map_size=214748364800)
anno = './dataset/ytbb_vid_train_filtered.json'
ytbb_interval = cfg.DATASET.YOUTUBEBB.FRAME_RANGE
vid_interval = cfg.DATASET.VID.FRAME_RANGE

lmdb_dataset = VIDYTBBLMDB(cfg, db, anno, ytbb_interval, vid_interval)
train_loader = DataLoader(lmdb_dataset,
                          batch_size=cfg.TRAIN.BATCH_SIZE,
                          num_workers=cfg.TRAIN.NUM_WORKERS,
                          pin_memory=True)
batch_num = len(train_loader)
logger.info('Finish building dataset')

# load pretrained backbone weights
if cfg.BACKBONE.PRETRAINED:
    logger.info('Load pretrained backbone')
    pretrained_path = cfg.BACKBONE.PRETRAINED
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
                    map_location=lambda storage, loc: storage.cuda(device))
    # saimrpn.backbone.load_state_dict(pretrained_dict, strict=False)
    load_pretrained_and_check(siamrpn.backbone, pretrained_dict)

# firstly keep parameters of all backbone layers fixed
for param in siamrpn.backbone.parameters():
    param.requires_grad = False
for m in siamrpn.backbone.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

optimizer = build_SGD(cfg, siamrpn)
scheduler = LogScheduler(optimizer,
                         start_lr=cfg.TRAIN.LR.START,
                         end_lr=cfg.TRAIN.LR.END,
                         epochs=cfg.TRAIN.EPOCH)

# train
logger.info('Start training')
for epoch in range(1, cfg.TRAIN.EPOCH+1):

    # unfreeze last two layers when reach setting
    if epoch == cfg.BACKBONE.TRAIN_EPOCH:
        logger.info('Start train backbone')
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(siamrpn.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(siamrpn.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        optimizer = build_SGD(cfg, siamrpn)

    data_end = time.time()
    for idx, data in enumerate(tqdm(train_loader)):
        data_time = time.time() - data_end
        forward_end = time.time()
        template = data['template'].cuda()    # (N,3,127,127)
        search = data['search'].cuda()        # (N,3,255,255)
        target_cls = data['label_cls'].cuda()
        target_reg = data['label_loc'].cuda()
        reg_weight = data['label_loc_weight'].cuda()
        pred_cls, pred_reg = siamrpn(template, search)
        forward_time = time.time() - forward_end
        # (N,2K,17,17) (N,4K,17,17)
        cls_loss = cls_cross_entropy_loss(pred_cls, target_cls)
        reg_loss = reg_smooth_l1_loss(pred_reg, target_reg, reg_weight)
        total_loss = cls_loss + cfg.TRAIN.LAMBDA * reg_loss

        # write losses
        tb_idx = (epoch - 1) * batch_num + idx
        tb_writer.add_scalar('losses/cls_loss', cls_loss.item(), tb_idx)
        tb_writer.add_scalar('losses/reg_loss', reg_loss.item(), tb_idx)
        tb_writer.add_scalar('losses/total_loss', total_loss.item(), tb_idx)
        tb_writer.add_scalar('time/data', data_time, tb_idx)
        tb_writer.add_scalar('time/forward', forward_time, tb_idx)
        for i, pg in enumerate(optimizer.param_groups):
            tb_writer.add_scalar('lr/group{}'.format(i + 1), pg['lr'], tb_idx)


        # optim
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(siamrpn.parameters(), cfg.TRAIN.GRAD_CLIP)
        optimizer.step()
    
        data_end = time.time()

    scheduler.step()
    # each epoch save model state_dict
    if not os.path.exists(cfg.TRAIN.SAVE_DIR):
        os.makedirs(cfg.TRAIN.SAVE_DIR)
    torch.save({
        'epoch': epoch,
        'model': siamrpn.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, cfg.TRAIN.SAVE_DIR+'/checkpoint_e%d.pth' % (epoch))
    logger.info('save model at'+cfg.TRAIN.SAVE_DIR+'/checkpoint_e%d.pth' % (epoch))

logger.info('Finish training!')







