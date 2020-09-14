"""
In this script, we will transform raw .jpg images
to lmdb format dataset to speed up dataload.
So what we need to do?
1. read the anno json file(yt_bb & VID)
2. filter out substandard videos/tracks/frames
3. construct key-value pair to be saved as
   lmdb file, key is each frame's relative path,
   value is encoded image
"""
# TODO: strengthen filter condition
import os
import sys
import cv2
import json
import lmdb
import hashlib
import functools
import numpy as np
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from config.config import cfg
from multiprocessing import Pool
# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

# your pysot location
yt_bb_json = cfg.DATASET.VID.ANNO
vid_json = cfg.DATASET.YOUTUBEBB.ANNO
path_format = '{}.{}.{}.jpg'

def filter_zero(anno_data):
    anno_data_new = {}
    for video, tracks in anno_data.items():
        new_tracks = {}
        for trk, frames in tracks.items():
            new_frames = {}
            for frm, bbox in frames.items():
                if not isinstance(bbox, dict):
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = bbox
                    if w <= 0 or h <= 0:
                        continue
                new_frames[frm] = bbox
            if len(new_frames) > 0:
                new_tracks[trk] = new_frames
        if len(new_tracks) > 0:
            anno_data_new[video] = new_tracks
    return anno_data_new

def worker(video_name):
    if 'ILSVRC2015' in video_name:
        video_path = os.path.join(cfg.DATASET.VID.ROOT, video_name)
    else:
        video_path = os.path.join(cfg.DATASET.YOUTUBEBB.ROOT, video_name)
    kv = {}
    for track in list(ytbb_vid_anno[video_name].keys()):
        for frame in list(ytbb_vid_anno[video_name][track]):
            if frame == 'frames':
                continue
            else:
                img =  cv2.imread(os.path.join(video_path, path_format.format(frame, track, 'x')))
            img_encode = cv2.imencode('.jpg', img)[1]
            img_encode = img_encode.tobytes()
            image_name = os.path.join(video_name, path_format.format(frame, track, 'x'))
            kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv

def create_lmdb(video_names, output_dir='./', num_threads=mp.cpu_count()):

    tic = cv2.getTickCount()
    db = lmdb.open(output_dir, map_size=int(200e9))
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)
    toc = cv2.getTickCount()
    print("create lmdb cost {} seconds".format((toc-tic)/cv2.getTickFrequency()))

with open(yt_bb_json, 'r') as yt_bb:
    yt_bb_anno = json.load(yt_bb)  # dict

with open(vid_json, 'r') as vid:
    vid_anno = json.load(vid)      # dict

# merge two datasets' anno
vid_anno.update(yt_bb_anno)

# filter out substandard videos/tracks/frames
anno_new = filter_zero(vid_anno)

# delete those tracks which has no frames and add n new key 'frames' under track
for video in list(anno_new.keys()):
    for track in anno_new[video]:
        frames = anno_new[video][track]
        frames = list(map(int,
                      filter(lambda x: x.isdigit(), frames.keys())))
        frames.sort()
        anno_new[video][track]['frames'] = frames
        if len(frames) <= 0:
            print("{}/{} has no frames".format(video, track))
            del anno_new[video][track]

# delete those videos which has no tracks
for video in list(anno_new.keys()):
    if len(anno_new[video]) <= 0:
        print("{} has no tracks".format(video))
        del anno_new[video]

# save the filtered anno json
ytbb_vid_train_filtered = './ytbb_vid_train_filtered.json'
json.dump(anno_new, open(ytbb_vid_train_filtered, 'w'), indent=4, sort_keys=True)
print("New anno json of ytbb_vid has saved successfully!")

with open(ytbb_vid_train_filtered, 'r') as f:
    ytbb_vid_anno = json.load(f)      # dict

print("Whole dataset has {} videos".format(len(ytbb_vid_anno)))
video_names = list(ytbb_vid_anno.keys())  # test a small example
# print(video_names)
create_lmdb(video_names)