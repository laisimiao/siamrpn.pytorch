# input: bbox, ft_size, neg, anchor
# output: cls_label, reg_label
import math
import numpy as np
from config.config import cfg
from utils.coordinate import center2corner, corner2center, IoU

def scatter_anchors(stride, ratios, scales, x_c, ft_size):
    """
    Args:
        stride: total stride
        ratios: ratios of anchors
        scales: scales of anchors
        x_c: detection frame center distance
        ft_size: feature map size

    Returns:
        (K * ft_size * ft_size) anchors scatter
        in detection frame (xyxy, xywh) format
    """
    K = len(ratios) * len(scales)  # anchor number
    anchors = np.zeros((K, 4), dtype=np.float32)  # (K, 4)
    size = stride * stride
    count = 0
    for r in ratios:
        ws = int(math.sqrt(size * 1. / r))
        hs = int(ws * r)

        for s in scales:
            w = ws * s
            h = hs * s
            anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
            count += 1

    top_left = x_c - ft_size // 2 * stride
    ori = np.array([top_left] * 4, dtype=np.float32)
    zero_anchors = anchors + ori  # (K, 4) xyxy format

    x1 = zero_anchors[:, 0]
    y1 = zero_anchors[:, 1]
    x2 = zero_anchors[:, 2]
    y2 = zero_anchors[:, 3]

    x1, y1, x2, y2 = map(lambda x: x.reshape(K, 1, 1), [x1, y1, x2, y2])
    cx, cy, w, h = corner2center([x1, y1, x2, y2])  # (K, 1, 1)

    disp_x = np.arange(0, ft_size).reshape(1, 1, -1) * stride
    disp_y = np.arange(0, ft_size).reshape(1, -1, 1) * stride

    cx = cx + disp_x  # (K, 1, 1) + (1, 1, 17) -> (K, 1, 17)
    cy = cy + disp_y  # (K, 1, 1) + (1, 17, 1) -> (K, 17, 1)

    # broadcast
    zero = np.zeros((K, ft_size, ft_size), dtype=np.float32)
    cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])  # (K, 17, 17)
    x1, y1, x2, y2 = center2corner([cx, cy, w, h])          # (K, 17, 17)

    all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                   np.stack([cx, cy, w, h]).astype(np.float32))
    # (4, K, 17, 17), (4, K, 17, 17)
    return all_anchors

def generate_track_all_anchor(stride, ratios, scales, score_size):
    # This method generate anchors for track phase need
    K = len(ratios) * len(scales)  # anchor number
    anchor = np.zeros((K, 4), dtype=np.float32)  # (K, 4)
    size = stride * stride
    count = 0
    for r in ratios:
        ws = int(math.sqrt(size * 1. / r))
        hs = int(ws * r)

        for s in scales:
            w = ws * s
            h = hs * s
            anchor[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
            count += 1

    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
    total_stride = stride
    anchor_num = anchor.shape[0]
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    # 默认以图像中心为原点
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def generate_labels(all_anchors, target_bbox):
    """
    There is no negative pairs in SiamRPN
    Args:
        anchors: all anchors scattered in detection frame
                 from `scatter_anchors` method
        bbox: ground truth bbox utils in detection
              frame, xyxy format

    Returns:
        cls: (K, 17, 17) cls branch label
        delta: (4, K, 17, 17) reg branch label
        delta_weight: (K, 17, 17) reg loss normalization factor
    """
    K, ft_size = all_anchors[0].shape[1], all_anchors[0].shape[2]
    tcx, tcy, tw, th = corner2center(target_bbox)

    # -1 ignore 0 negative 1 positive
    cls = -1 * np.ones((K, ft_size, ft_size), dtype=np.int64)
    delta = np.zeros((4, K, ft_size, ft_size), dtype=np.float32)
    delta_weight = np.zeros((K, ft_size, ft_size), dtype=np.float32)

    anchor_corner, anchor_center = all_anchors[0], all_anchors[1]

    x1, y1, x2, y2 = anchor_corner[0], anchor_corner[1], \
                     anchor_corner[2], anchor_corner[3]

    cx, cy, w, h   = anchor_center[0], anchor_center[1], \
                     anchor_center[2], anchor_center[3]
    # delta: (4, K, 17, 17) broadcast
    delta[0] = (tcx - cx) / w
    delta[1] = (tcy - cy) / h
    delta[2] = np.log(tw / w)
    delta[3] = np.log(th / h)
    # calculate IoU between target_bbox and all anchors
    overlap = IoU([x1, y1, x2, y2], target_bbox)

    pos = np.where(overlap > cfg.TRAIN.THR_HIGH)
    neg = np.where(overlap < cfg.TRAIN.THR_LOW)

    def select(position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num

    # make sure at most 16 positive samples and totally 64 sam-
    # ples from one training pair as paper said
    pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
    neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

    # print("pos_num: ", pos_num)
    cls[pos] = 1
    cls[neg] = 0
    delta_weight[pos] = 1. / (pos_num + 1e-6)  # avoid ZeroDivisionError
    # (K,17,17) (4,K,17,17) (K,17,17) (1,)
    return cls, delta, delta_weight
