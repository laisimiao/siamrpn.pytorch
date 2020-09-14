# Use got10k toolkit run on OTB benchmark
import torch
import argparse
from model import SiamRPN
from track import SiamRPNTracker
from got10k.experiments import ExperimentOTB
from utils.load_state import load_pretrained_and_check

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('-root', '--root_dir', default='', type=str,
                    help='your path to raw OTB dataset')
args = parser.parse_args()

root_dir = args.root_dir
model_path = './checkpoints/checkpoint_e50.pth'
# initialize custom tracker
model = SiamRPN(anchor_num=5)
device = torch.cuda.current_device()
pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))
load_pretrained_and_check(model, pretrained_dict['model'])

tracker = SiamRPNTracker(model.cuda())
experiment = ExperimentOTB(root_dir)
experiment.run(tracker)
experiment.report([tracker.name])
