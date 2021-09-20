import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
#from lightning_train1 import FusionModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import cv2
import git
import os, argparse, glob, random, tempfile
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
plt = matplotlib.pyplot
from copy import deepcopy
from scipy.special import softmax
import pandas as pd
import numpy as np
import cv2

# Torch imports
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

# Custom imports
import git
import sys
repo = '/home/idris/robosurgery/deepStitch2'
repo = git.Repo(repo, search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
from convlstmcells import ConvLSTMCell, ConvTTLSTMCell
import settings


class FusionModel(LightningModule):
    def __init__(self, args):
        super(FusionModel, self).__init__()
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.hparams = args
        self.actual = [];
        self.actual_train = []
        self.predicted = [];
        self.predicted_train = []
        self.predicted_softmax = [];
        self.predicted_softmax_train = []
        self.batch_size = self.hparams.batch_size
        ############################################################################################
        # Generate the tra300in and test splits
        fns = []
        for class_curr in self.hparams.include_classes:
            fns.extend(glob.glob(os.path.join(self.hparams.datadir, class_curr, '%s*' % self.hparams.flow_method)))
        idx = list(range(len(fns)))
        random.seed(self.hparams.seed);
        random.shuffle(idx)
        self.hparams.idx_train = idx[:int(self.hparams.train_proportion * len(idx))].copy()  # Save as hyperparams
        self.hparams.idx_test = idx[int(self.hparams.train_proportion * len(idx)):].copy()  # Save as hyperparams
        ############################################################################################

        # Model specific
        original_model_rgb = models.__dict__[args.arch](pretrained=self.hparams.use_pretrained)
        original_model_of = models.__dict__[args.arch](pretrained=self.hparams.use_pretrained)

        self.hidden_size = args.hidden_size
        self.num_classes = len(args.include_classes)
        self.fc_size = args.fc_size
        self.trainable_base = args.trainable_base

        nF = 7
        # select a base model
        if args.arch.startswith('alexnet'):
            nF = 6
            self.features_rgb = original_model_rgb.features
            for i, param in enumerate(self.features_rgb.parameters()):
                param.requires_grad = self.trainable_base
            self.features_of = original_model_of.features
            for i, param in enumerate(self.features_of.parameters()):
                param.requires_grad = self.trainable_base

            # Make the output of each one be to fc_size/2 so that we cooncat the two fc outputs
            self.fc_pre_rgb = nn.Sequential(nn.Linear(256 * nF * nF, int(args.fc_size / 2)),
                                            nn.Dropout()) if 'conv' not in args.rnn_model else None
            self.fc_pre_of = nn.Sequential(nn.Linear(256 * nF * nF, int(args.fc_size / 2)),
                                           nn.Dropout()) if 'conv' not in args.rnn_model else None
            self.final_channels = 256

        elif args.arch.startswith('resnet18'):
            # self.features = nn.Sequential(*list(original_model.children())[:-2])
            # for i, param in enumerate(self.features.parameters()):
            #       param.requires_grad = self.trainable_base
            # self.fc_pre = nn.Sequential(nn.Linear(512*7*7, int(args.fc_size/2)), nn.Dropout()) if 'conv' not in args.rnn_model else None
            # self.final_channels = 512
            self.features_rgb = nn.Sequential(*list(original_model_rgb.children())[:-2])
            for i, param in enumerate(self.features_rgb.parameters()):
                param.requires_grad = self.trainable_base
            self.features_of = nn.Sequential(*list(original_model_of.children())[:-2])
            for i, param in enumerate(self.features_of.parameters()):
                param.requires_grad = self.trainable_base

            self.fc_pre_rgb = nn.Sequential(nn.Linear(512 * nF * nF, int(args.fc_size / 2)),
                                            nn.Dropout()) if 'conv' not in args.rnn_model else None
            self.fc_pre_of = nn.Sequential(nn.Linear(512 * nF * nF, int(args.fc_size / 2)),
                                           nn.Dropout()) if 'conv' not in args.rnn_model else None
            self.final_channels = 512

        else:
            raise ValueError('architecture base model not yet implemented choices: alexnet, vgg16, ResNet 18/34')
        # Select an RNN
        # print(self.features)
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=args.fc_size,
                               hidden_size=args.hidden_size,
                               num_layers=args.rnn_layers,
                               batch_first=True)
            self.fc = nn.Linear(args.hidden_size, self.num_classes)
        elif args.rnn_model == 'convLSTM':
            # Twice number of channels for RGB and OF which are concat
            self.rnn = ConvLSTMCell(input_channels=self.final_channels * 2,
                                    hidden_channels=int(self.hparams.hidden_size), kernel_size=3, bias=True)

            self.rnn1 = ConvLSTMCell(input_channels=640,
                                     hidden_channels=640, kernel_size=3, bias=True)

            self.fc = nn.Linear(int(self.hparams.hidden_size) * nF * nF, self.num_classes)
            self.fc_k = nn.Linear(int(self.hparams.hidden_size) * nF * nF,
                                  len(self.hparams.kinematics_features))  # replace with the dimensions of the kinematics data

            self.fc_attn1 = nn.Linear((int(self.hparams.hidden_size) + (2 * self.final_channels)) * nF * nF,
                                      ((int(self.hparams.hidden_size) + (2 * self.final_channels)) * nF * nF) // 2)

            self.fc_attn2 = nn.Linear(
                ((int(self.hparams.hidden_size) + (2 * self.final_channels)) * nF * nF) // 2,
                nF * nF)

            self.fc_attn3 = nn.Linear(
                int(self.hparams.hidden_size) * nF * nF,
                nF * nF)

            self.fc_conv1 = nn.Conv2d(in_channels=512,
                                      out_channels=512, kernel_size=3, padding=1)


        elif args.rnn_model == 'convttLSTM':
            # Twice number of channels for RGB and OF which are concat
            self.rnn = ConvTTLSTMCell(input_channels=self.final_channels * 2, hidden_channels=self.final_channels,
                                      order=3, steps=5, ranks=16, kernel_size=3, bias=True)

            self.fc = nn.Linear(self.final_channels * nF * nF, self.num_classes)
        else:
            raise ValueError('Not implemented, choose LSTM, convLSTM, convttLSTM type')

        self.modelName = '%s_%s_latefusion_trainbaseparams_%s' % (args.arch, args.rnn_model, args.trainable_base)


def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("could not open: ", video_path)
        return -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    return length, width, height


def visualize_att(alphas, image_path, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    # model1 = torch.load(alphas)
    # model.load_state_dict(torch.load(alphas))
    # model = copy.deepcopy(model1)

    checkpoint = torch.load(alphas, map_location=torch.device('cpu'))

    model = FusionModel()

    # print(checkpoint.keys())
    model1 = checkpoint['state_dict']

    model.load_state_dict(model1)
    model.eval()



    length, _, _ = video_frame_count(image_path)
    cap = cv2.VideoCapture(image_path)
    #q = queue.Queue(self.frames_num)
    frames = list()
    count = 0
    while count < length:
        ret, frame = cap.read()
        if type(frame) == type(None):
            break
        else:
            frames.append(frame)

    for i, frame in enumerate(frames):
        #image = Image.open(frame)
        #image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)


        plt.imshow(frame)
        current_alpha = model[i, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=36, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
    plt.imshow(alpha, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--img_path', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')

    args = parser.parse_args()

    visualize_att(args.model, args.img_path)