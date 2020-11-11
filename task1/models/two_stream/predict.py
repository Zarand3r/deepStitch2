#%%
import pandas as pd
import os
import argparse
import torch
import torchvision
import numpy as np
# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings1
import lightning_train as classifier

# batch = [data, labels]
# data = [sample_1, sample_2, ... sample_nbatch]
# sampe = [frame_1, frame_2, ... frame_3]
# frame = [HxWxC]

def apply_transforms_GPU(self, batch, random_crop = False):
        # Prepares the outputs
        nB, nF, nH, nW, nC = batch[0].size()
        if random_crop:
            rgb = self.augGPU_resize(batch[0][:, :, :, :int(nW/2), :].type(torch.float)/255., npix_resize = (240, 240), random_crop = True)
            of  = self.augGPU_resize(batch[0][:, :, :, int(nW/2):, :].type(torch.float)/255., npix_resize = (240, 240), random_crop = True)
        else:
            rgb = self.augGPU_resize(batch[0][:, :, :, :int(nW/2), :].type(torch.float)/255., npix_resize = (224, 224))
            of  = self.augGPU_resize(batch[0][:, :, :, int(nW/2):, :].type(torch.float)/255., npix_resize = (224, 224))

        #of = self.augGPU_normalize_inplace(of, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        #rgb = self.augGPU_normalize_inplace(rgb, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        #print(of.reshape(-1, 3).mean(0))
        rgb = self.augGPU_normalize_inplace(rgb, mean = [0.3, 0.2, 0.2], std = [0.2, 0.2, 0.2])
        of = self.augGPU_normalize_inplace(of, mean = [0.99, 0.99, 0.99], std = [0.005, 0.005, 0.005])
        #print(of.reshape(-1, 3).mean(0))
        #print(of.reshape(-1, 3).var(0))

        return [torch.stack([rgb, of], axis = -1), batch[1]]

def augGPU_resize(self, input, seed = None, npix_resize = (224, 224), random_crop = False):
    """ Resizing operation using interp so it is done on the GPU"""
    if len(input.size()) == 4: # Single batch
        input = input.unsqueeze(0)
    nB, nF, _, _, nC = input.size()

    if seed != None:
        random.seed(seed)

    resized_rgb = torch.zeros(nB, nF, npix_resize[0], npix_resize[1], nC).type_as(input)    
    for bb in range(nB):
        for ff in range(nF):
            im = input.permute(0, 1, 4, 2, 3)[0, 0, :, :, :].unsqueeze(0).type(torch.float)
            resized_rgb[bb, ff, :, :, :] = F.interpolate(im, size = npix_resize).permute(0,2,3,1)
    # Right now does random crop for each of optic flow and rgb
    if random_crop:
        x0 = random.randint(0, npix_resize[1]-225)
        y0 = random.randint(0, npix_resize[0]-225)
        resized_rgb = resized_rgb[:, :, y0:y0+224, x0:x0+224, :]
    
    return resized_rgb

def augGPU_normalize_inplace(self, input, mean = [0.3, 0.3, 0.3], std=[0.1, 0.1, 0.1]):
    """Does an in place normalization on GPU"""
    mean = torch.as_tensor(mean).type_as(input)
    std = torch.as_tensor(std).type_as(input)
    input = input.sub_(mean[None, None, None, None, :]).div_(std[None, None, None, None, :])
    return input


# use the CustomDataset class in lightning_train.py

# Load the preprocessed input video
# If it has more frames than the sliding window, iteratively slice and classify each slice
# Return a list of classifications per frame. The starting point of the slice that has a different label than before is the beginning of the new label.

def predict(hparams):
    # need to make input shape consistent with what is defined in forward function
    if torchvision.__version__[:3] == '0.4':
        video = torchvision.io.read_video(hparams.input_file)[0]
    else: # Newer version
        video = torchvision.io.read_video(hparams.input_file, pts_unit = 'sec')[0]
    #video = video.reshape((1,33, 540, 1920, 3,1))
    # look at dataloader and apply_transforms_GPU in lightning_train.py
    video = torch.reshape(video,(1,33,540,1920,3,1))
    video = video.float()


    # create a batch of (videos, label) tuples (can just be the one). The label is just 0, 1, 2 ...
    # feed in the input x to the prediction model as apply_transforms_GPU(batch, random_crap=hparams.random_crop)
    model = classifier.FusionModel(hparams)
    checkpoint = torch.load(hparams.checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    output = model(x)

    model = classifier.FusionModel.load_from_checkpoint(hparams.checkpoint_path)
    model.freeze()
    out = mode(x)
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input argument
    parser.add_argument('--input_file', default=f'{homedir}/task1/models/two_stream/test/test1.mp4', help='Input file to predict')
    parser.add_argument('--checkpoint_path', default=f'{homedir}/task1/models/two_stream/test/checkpoint.ckpt', help='path to load checkpoints')
    parser.add_argument('--loadchk', default='', help='Pass through to load training from a checkpoint')
    parser.add_argument('--datadir', default=settings1.data_directory, help='train directory')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device number')
    parser.add_argument('--arch', default='alexnet', help='model architecture')
    parser.add_argument('--trainable_base', default=0, type=int, help='Whether to train the feature extractor')
    parser.add_argument('--rnn_model', default='LSTM', type=str, help='RNN model at clasification')
    parser.add_argument('--rnn_layers', default=2, type=int, help='number of rnn layers')
    parser.add_argument('--hidden_size', default=16, type=int, help='output size of rnn hidden layers')
    parser.add_argument('--fc_size', default=32, type=int, help='size of fully connected layer before rnn')
    parser.add_argument('--epochs', default=60, type=int, help='manual epoch number')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--lr_lambdas', default=0.9, type=float, help='Schedulre hyperparam')
    parser.add_argument('--include_classes', default='positive_negative', type=str, help='Which classnames to include seperated by _ e.g. 00_01')
    parser.add_argument('--flow_method', default='flownet', type=str, help='Which flow method to use (flownet or dali)')
    parser.add_argument('--random_crop', default=0, type=int, help='Whether or not to augment with random crops...')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--train_proportion', default=0.8, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--accum_batches', default=3, type=int)
    parser.add_argument('--overfit', default=0, type=int)
    parser.add_argument('--auto_lr', default=0, type=int)
    parser.add_argument('--use_pretrained', default=1, type=int, help='whether or not to load pretrained weights')
    parser.add_argument('--logging_dir', default='lightning_logs', type=str)
    parser.add_argument('--loader_nframes', default=140, type=int, help='How many frames to load at stride 2')
    parser.add_argument('--loader_stride', default=2, type=int, help='stride for dataloader')
    parser.add_argument('--number_workers', default=0, type=int, help='number of workers for Dataloader')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--masked', dest='masked', action='store_true', help = "train on masked?")
    
    hparams = parser.parse_args()
    hparams.trainable_base = True if hparams.trainable_base == 1 else False
    hparams.random_crop = True if hparams.random_crop == 1 else False
    hparams.auto_lr = True if hparams.auto_lr == 1 else False
    hparams.use_pretrained = True if hparams.use_pretrained == 1 else False
    if hparams.masked:
        hparams.masked = "_masked"
    else:
        hparams.masked = ""

    classification_name = "positive_negative"
    if hparams.include_classes == '':
        raise ValueError('Please define the classes to use using the 00_01 underscore notation')
    else:
        classification_name = hparams.include_classes
        hparams.include_classes = hparams.include_classes.split('_')

    predict(hparams)
    

