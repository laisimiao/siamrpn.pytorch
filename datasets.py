import cv2
import json
import hashlib
import numpy as np
from utils.coordinate import Center, center2corner
from augmentation import Augmentation
from torch.utils.data.dataset import Dataset
from label import scatter_anchors, generate_labels

class VIDYTBBLMDB(Dataset):
    def __init__(self, cfg, db, anno, ytbb_interval, vid_interval):
        super(VIDYTBBLMDB, self).__init__()
        self.cfg = cfg
        self.txn = db.begin(write=False)
        with open(anno, 'r') as f:
            self.ytbb_vid_anno = json.load(f)  # dict
        self.num_video = len(self.ytbb_vid_anno)
        self.video_names = list(self.ytbb_vid_anno.keys())
        self.ytbb_interval = ytbb_interval
        self.vid_interval = vid_interval

        # data augmentation
        self.template_aug = Augmentation(
            self.cfg.DATASET.TEMPLATE.SHIFT,
            self.cfg.DATASET.TEMPLATE.SCALE,
            self.cfg.DATASET.TEMPLATE.BLUR,
            self.cfg.DATASET.TEMPLATE.FLIP,
            self.cfg.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            self.cfg.DATASET.SEARCH.SHIFT,
            self.cfg.DATASET.SEARCH.SCALE,
            self.cfg.DATASET.SEARCH.BLUR,
            self.cfg.DATASET.SEARCH.FLIP,
            self.cfg.DATASET.SEARCH.COLOR
        )


    def __len__(self):
        return self.num_video

    def __getitem__(self, index):
        """
        Args:
            index: the video index

        Returns:
            template  frame(after augmentation)
            detection frame(after augmentation)
            cls_target
            reg_target

        """
        # TODO:analyse dataload time and transform label generate
        # TODO:to GPU
        video_name = self.video_names[index]
        video = self.ytbb_vid_anno[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        if 'ILSVRC2015' in video_name:
            self.frame_range = self.vid_interval
        else:
            self.frame_range = self.ytbb_interval
        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        template_image, search_image, template_box, search_box= self._decode(
            video_name, track, template_frame, search_frame)

        # get bounding box(has mapped to scaled level)
        template_box = self._get_bbox(template_image, template_box)
        search_box = self._get_bbox(search_image, search_box)

        # augmentation
        gray = self.cfg.DATASET.GRAY and self.cfg.DATASET.GRAY > np.random.random()
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        self.cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       self.cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels
        all_anchors = scatter_anchors(stride=self.cfg.ANCHOR.STRIDE,
                        ratios=self.cfg.ANCHOR.RATIOS,
                        scales=self.cfg.ANCHOR.SCALES,
                        x_c=self.cfg.TRAIN.SEARCH_SIZE//2,
                        ft_size=self.cfg.TRAIN.OUTPUT_SIZE)

        cls, delta, delta_weight = generate_labels(
            all_anchors, bbox)

        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight
        }

    def _decode(self, video_name, track, template_frame, search_frame):
        path_format = '{}.{}.{}.jpg'
        template_frame = "{:06d}".format(template_frame)
        search_frame = "{:06d}".format(search_frame)
        template_name = video_name + '/' + path_format.format(template_frame, track, 'x')
        search_name = video_name + '/' + path_format.format(search_frame, track, 'x')
        template_key = hashlib.md5(template_name.encode()).digest()
        search_key = hashlib.md5(search_name.encode()).digest()
        template_buffer = self.txn.get(template_key)
        search_buffer = self.txn.get(search_key)
        template_buffer = np.frombuffer(template_buffer, np.uint8)
        search_buffer = np.frombuffer(search_buffer, np.uint8)
        template_image = cv2.imdecode(template_buffer, cv2.IMREAD_COLOR)
        search_image = cv2.imdecode(search_buffer, cv2.IMREAD_COLOR)
        template_box = self.ytbb_vid_anno[video_name][track][template_frame] # xyxy
        search_box = self.ytbb_vid_anno[video_name][track][search_frame] # xyxy
        return template_image, search_image, template_box, search_box

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        # 这里的bbox是cropped_x上的坐标了，不再是raw image上的原本坐标了
        return bbox