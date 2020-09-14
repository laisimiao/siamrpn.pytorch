import cv2
import torch
import numpy as np
from config.config import cfg
import torch.nn.functional as F
from got10k.trackers import Tracker
from utils.coordinate import crop_like_SiamFC
from label import generate_track_all_anchor, scatter_anchors

cfg.merge_from_file('./config/config.yaml')
cfg.freeze()

class SiamRPNTracker(Tracker):
    """
    inherit Tracker of GOT10K toolkit and run benchmark
    """
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__(
            name='SiamRPNTracker'
        )

        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)

        self.anchors = scatter_anchors(cfg.ANCHOR.STRIDE,
                                       cfg.ANCHOR.RATIOS,
                                       cfg.ANCHOR.SCALES,
                                       cfg.TRACK.INSTANCE_SIZE // 2,
                                       self.score_size)[1]
        # 这样生成anchor也是可以的，只是下面106-107行删去，171行删去
        # self.anchors = generate_track_all_anchor(cfg.ANCHOR.STRIDE,
        #                                cfg.ANCHOR.RATIOS,
        #                                cfg.ANCHOR.SCALES,
        #                                self.score_size)  # (4, K, 17, 17)

        self.model = model
        self.model.eval()

    def init(self, image, box):
        """
        init tracker fix template feature
        Args:
            img(np.ndarray): BGR image
            box: (x, y, w, h) bbox
        Returns:
            no return
        """
        # convert image from Image to ndarray
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        self.center_pos = np.array([box[0]+(box[2]-1)/2,
                                    box[1]+(box[3]-1)/2]) # (cx,cy)
        self.size = np.array([box[2], box[3]])            # (w, h)

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = crop_like_SiamFC(image, self.center_pos, s_z,
                                  cfg.TRACK.EXEMPLAR_SIZE,
                                  self.channel_average)
        z_crop = torch.from_numpy(z_crop.astype(np.float32))
        z_crop = z_crop.unsqueeze(dim=0).permute(0,3,1,2)
        if cfg.CUDA:
            z_crop = z_crop.cuda()
        self.model.template(z_crop)

    def update(self, image):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
            best_score: selected bbox probability
        """

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = crop_like_SiamFC(image, self.center_pos, s_x,
                                  cfg.TRACK.INSTANCE_SIZE,
                                  self.channel_average)

        x_crop = torch.from_numpy(x_crop.astype(np.float32))
        x_crop = x_crop.unsqueeze(dim=0).permute(0,3,1,2)
        if cfg.CUDA:
            x_crop = x_crop.cuda()

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['reg'], self.anchors)

        # 这里因为我生成anchor的方法与pysot里面不太一样
        pred_bbox[0, :] -= cfg.TRACK.INSTANCE_SIZE // 2
        pred_bbox[1, :] -= cfg.TRACK.INSTANCE_SIZE // 2

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        # best_score = score[best_idx]
        return bbox  # must be (x,y,w,h) format

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _convert_bbox(self, delta, anchor):
        # print("delta shape: ", delta.shape)    torch.Size([1, 20, 17, 17])
        # print("anchor shape: ", anchor.shape)  (4, 5, 17, 17)


        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        # not (4, 5*17*17) 而是 (4, 17*17*5) 这里一定要注意view的用法
        delta = delta.data.cpu().numpy()

        # 这样才把anchor转化为(17*17*5,4)
        anchor = anchor.transpose((1,2,3,0)).reshape((-1, 4))

        # z这里我的anchor生成的方法和pysot测试阶段的anchors产生不一样
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

