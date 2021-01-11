#%%
import pandas as pd
import os
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# Custom imports
import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings1
import lightning_train as classifier
import salient_lightning_train as salient_classifier

# def val_dataloader(self):
#     val_dataset = CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_test , include_classes = self.hparams.include_classes, 
#                         flow_method = self.hparams.flow_method, balance_classes=True, mode = 'val', max_frames = self.hparams.loader_nframes,
#                         stride = self.hparams.loader_stride, masked = self.hparams.masked)
#     val_dataloader  = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.hparams.number_workers, drop_last=True)
#     return val_dataloader

# Also make an alternative method using a function to preprocess a single image, so we dont need to make a dataloader and dataset
def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    print(model)
    # Make input tensor require gradient
    X.requires_grad_()
    saliency = None
    
    # Forward pass
    scores = model(X)
    scores = torch.stack(scores)
    scores = torch.squeeze(scores, 1)
    y = torch.tensor([y]*len(scores))
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    print("Score of the label classification for each frame: ", scores)
    scores.backward(torch.ones(scores.size()))
    saliency = X.grad
    saliency = saliency.abs()
    saliency, _= torch.max(saliency, dim=4)
    saliency = torch.squeeze(saliency)
    print(saliency.shape)
    
    return saliency


def show_saliency_maps(args):
    model = salient_classifier.FusionModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams_file=args.hparams_path)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    loader = model.val_dataloader()
    i = 0
    dataiter = iter(loader)
    batch = dataiter.next()
    #sample = batch[0]
    X_tensor, y_tensor = model.apply_transforms_GPU(batch, random_crop=model.hparams.random_crop)
    # Compute saliency maps for images in X
    print("Input shape (nB, nF, nH, nW, nC, [rgb, of]]): ", X_tensor.shape)
    print("label shape: ", y_tensor.shape)
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    
    X = torch.squeeze(X_tensor)
    X = X.detach().numpy()
    X_rgb = X[:, :, :, :, 0]
    # Convert the saliency map from Torch Tensor to numpy array and show images and saliency maps together.
    saliency = saliency.numpy()
    saliency_rgb = saliency[:, :, :, 0]
    N = X_rgb.shape[0] #len(X)
    frames = [0, 6, 12, 18]
    for i in range(len(frames)):
        plt.subplot(2, len(frames), i + 1)
        plt.imshow(X_rgb[frames[i]])
        plt.axis('off')
        plt.subplot(2, len(frames), len(frames) + i + 1)
        plt.imshow(saliency_rgb[frames[i]], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    #plt.show()
    #to view images on ssh, use eog saliency.png
    plt.savefig("saliency_rgb.png")

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
    #predict(args)
    show_saliency_maps(args)
    

