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

# def val_dataloader(self):
#     val_dataset = CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_test , include_classes = self.hparams.include_classes, 
#                         flow_method = self.hparams.flow_method, balance_classes=True, mode = 'val', max_frames = self.hparams.loader_nframes,
#                         stride = self.hparams.loader_stride, masked = self.hparams.masked)
#     val_dataloader  = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.hparams.number_workers, drop_last=True)
#     return val_dataloader

def predict(args):
    model = classifier.FusionModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams_file=args.hparams_path)
    #model = classifier.FusionModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    model.eval()
    loader = model.val_dataloader()
    i = 0
    for batch in loader:
        i += 1
        if i >= 2:
            break
        #sample = batch[0]
        label = batch[1]
        input_cuda, target_cuda = model.apply_transforms_GPU(batch, random_crop=model.hparams.random_crop)
        prediction = model(input_cuda)
        print(prediction)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input argument
    parser.add_argument('--input_file', default=f'{homedir}/task1/models/two_stream/test/test1.mp4', help='Input file to predict')
    parser.add_argument('--checkpoint_path', default=f'{settings1.checkpoints}/two_stream/AC_CE_EF_FG/_ckpt_epoch_46.ckpt', help='path to load checkpoints')
    parser.add_argument('--hparams_path', default=f'{homedir}/task1/models/two_stream/test/hparams.yaml', help='path to load hyperparameters')
    args = parser.parse_args()
    predict(args)
    

