#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="TCN_data")
parser.add_argument('--split', default='1')
parser.add_argument('--stages', default=4)
parser.add_argument('--layers', default=10)
parser.add_argument('--f_maps', default=64)
parser.add_argument('--lr', default=0.0005)

args = parser.parse_args()

num_stages = args.stages
num_layers = args.layers
num_f_maps = args.f_maps
features_dim = 2048
bz = 1
lr = args.lr
num_epochs = 50

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "TCN_data":
    sample_rate = 1

vid_list_file = "/mnt/md1/richard_bao/balint_data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "/mnt/md1/richard_bao/balint_data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "/mnt/md1/richard_bao/balint_data/"+args.dataset+"/features/"
gt_path = "/mnt/md1/richard_bao/balint_data/"+args.dataset+"/groundTruth/"

mapping_file = "/mnt/md1/richard_bao/balint_data/"+args.dataset+"/mapping.txt"

model_dir = "models/"+args.dataset+"/split_"+args.split
results_dir = "results/"+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
# actions = file_ptr.read().split('\n')
# actions.remove('')
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
