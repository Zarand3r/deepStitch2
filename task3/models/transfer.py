import os, argparse, glob, random, tempfile
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.special import softmax
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
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings3
import test as feature_classifier
from lightning_train import CustomDataset

class TransferLearning(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        fns = []
        for class_curr in self.hparams.include_classes:
            fns.extend(glob.glob(os.path.join(self.hparams.datadir, class_curr, '%s*' % self.hparams.flow_method)))
        idx = list(range(len(fns)))
        random.seed(self.hparams.seed); random.shuffle(idx)
        self.hparams.filenames = []
        self.hparams.idx_train  = idx[:int(self.hparams.train_proportion*len(idx))].copy() # Save as hyperparams
        self.hparams.idx_test   = idx[int(self.hparams.train_proportion*len(idx)):].copy() # Save as hyperparams
        # init a pretrained resnet
        backbone = feature_classifier.FusionModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
        print("BACKBONE: ", backbone)
        self.model = backbone
        num_filters = backbone.fc.in_features
        print("NUM_FILTERS: ", num_filters)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)
        print("LAYERS: ", layers)
        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = len(args.include_classes)
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        print(x.shape)
        print(self.feature_extractor)
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x)
            #representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(1)
        return x, _, _


    def training_step(self, batch, batch_idx):
            # Batch is already on GPU by now
            input_cuda, target_cuda = self.apply_transforms_GPU(batch, random_crop=self.hparams.random_crop)
            output, _, _ = self(input_cuda)
            output = output[:, -1, :]
            loss = F.cross_entropy(output, target_cuda.type(torch.long))
            self.actual_train.append(target_cuda.item())
            self.predicted_train.append(output.topk(1,1)[-1].item())
            self.predicted_softmax_train.append(softmax(output.detach().cpu().numpy(), axis = -1)) # Save scores
            tensorboard_logs = {'train/loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
            input_cuda, target_cuda = self.apply_transforms_GPU(batch, random_crop=False)
            output, _, _ = self(input_cuda)
            output = output[:, -1, :]
            loss = F.cross_entropy(output, target_cuda.type(torch.long))
            self.actual.append(target_cuda.item())
            self.predicted.append(output.topk(1,1)[-1].item())
            if self.actual[-1] != self.predicted[-1]:
                    print("MISCLASSIFIED: ", batch_idx) # print self.filenames[batch_idx][0]
            self.predicted_softmax.append(softmax(output.detach().cpu().numpy(), axis = -1)) # Save scores

            return {'val_loss': loss}

    def validation_end(self, outputs):
            ########### Log to tensorboard#######################################################
            if self.trainer.overfit_pct==0: # Not a debug loop, otherwise roc_auc can throw errors
                    if self.num_classes ==2:
                            auc_train       = roc_auc_score(self.actual_train, np.vstack(self.predicted_softmax_train)[:, 1])
                            auc_val         = roc_auc_score(self.actual, np.vstack(self.predicted_softmax)[:, 1])
                    else:
                            auc_train       = roc_auc_score(self.actual_train, np.vstack(self.predicted_softmax_train), multi_class = 'ovr')
                            auc_val         = roc_auc_score(self.actual, np.vstack(self.predicted_softmax), multi_class = 'ovr')
            else:
                    auc_train, auc_val = 0, 0
            
            top1_val = self.send_im_calculate_top1(self.actual, self.predicted, cmap_use = 'Blues', name = 'val/conf_mat')
            top1_train = self.send_im_calculate_top1(self.actual_train, self.predicted_train, cmap_use = 'Oranges', name = 'train/conf_mat')
            #####################################################################################
            self.actual = []; self.actual_train = []
            self.predicted = []; self.predicted_train = []
            self.predicted_softmax = []; self.predicted_softmax_train = []
            
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val/loss': avg_loss, 'val/top1': top1_val, 'val/auc': auc_val, 'train/top1': top1_train, 'train/auc': auc_train, 'step': self.current_epoch}
            return {'val_loss': avg_loss, 'val_acc':torch.tensor(top1_val), 'log': tensorboard_logs}
    
    def send_im_calculate_top1(self, actual, predicted, cmap_use = 'Blues', name = 'tmp/name'):
            self.model.send_im_calculate_top1(self, actual, predicted, cmap_use = 'Blues', name = 'tmp/name')
            
    def configure_optimizers(self):
            self.model.configure_optimizers()

    # def send_im_calculate_top1(self, actual, predicted, cmap_use = 'Blues', name = 'tmp/name'):
    #         cm = confusion_matrix(actual, predicted)
    #         fig = plt.figure(); sns.heatmap(cm, cmap = cmap_use, ax =plt.gca(), annot = True, xticklabels = self.hparams.include_classes, yticklabels = self.hparams.include_classes)
    #         self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch, close = True)
    #         top1 = float(sum([a==b for a,b in zip(actual, predicted)]))/len(actual)
    #         return top1
            
    # def configure_optimizers(self):
    #         self.layers_to_fit = [{'params': self.fc.parameters()}, {'params': self.rnn.parameters()}]
    #         if self.fc_pre_rgb != None:
    #                 self.layers_to_fit.append({'params': self.fc_pre_rgb.parameters()})
    #                 self.layers_to_fit.append({'params': self.fc_pre_of.parameters()})
    #         if self.trainable_base:
    #                 self.layers_to_fit.append({'params': self.features_rgb.parameters()})
    #                 self.layers_to_fit.append({'params': self.features_of.parameters()})
            
    #         optimizer = torch.optim.Adam(self.layers_to_fit,
    #                                                         lr=self.hparams.lr, betas=(0.9, 0.999), 
    #                                                         weight_decay = self.hparams.weight_decay)
    
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.1)
    #         return [optimizer], [scheduler]

    def train_dataloader(self):
            train_dataset   = CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_train , include_classes = self.hparams.include_classes, 
                                                    flow_method = self.hparams.flow_method, balance_classes=True, mode = 'train', max_frames = self.hparams.loader_nframes,
                                                    stride = self.hparams.loader_stride, masked = self.hparams.masked)
            self.filenames = train_dataset.filtered_fns
            train_dataloader        = DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.number_workers, drop_last=True)
            self.epoch_len = len(train_dataset)
            return train_dataloader

    def val_dataloader(self):
            val_dataset     = CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_test , include_classes = self.hparams.include_classes, 
                                                    flow_method = self.hparams.flow_method, balance_classes=True, mode = 'val', max_frames = self.hparams.loader_nframes,
                                                    stride = self.hparams.loader_stride, masked = self.hparams.masked)
            val_dataloader  = DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.number_workers, drop_last=True)
            return val_dataloader

    def apply_transforms_GPU(self, batch, random_crop = False):
            # Prepares the outputs
            nB, nF, nH, nW, nC = batch[0].size()
            if random_crop:
                    rgb = self.augGPU_resize(batch[0][:, :, :, :int(nW/2), :].type(torch.float)/255., npix_resize = (240, 240), random_crop = True)
                    of      = self.augGPU_resize(batch[0][:, :, :, int(nW/2):, :].type(torch.float)/255., npix_resize = (240, 240), random_crop = True)
            else:
                    rgb = self.augGPU_resize(batch[0][:, :, :, :int(nW/2), :].type(torch.float)/255., npix_resize = (224, 224))
                    of      = self.augGPU_resize(batch[0][:, :, :, int(nW/2):, :].type(torch.float)/255., npix_resize = (224, 224))

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
                            im = input.permute(0, 1, 4, 2, 3)[bb, ff, :, :, :].unsqueeze(0).type(torch.float)
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

def predict(args):
    #model = classifier.FusionModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams_file=args.hparams_path)
    backbone = classifier.FusionModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    #backbone.freeze()
    print(backbone)
    num_filters = backbone.fc.in_features
    print(num_filters)
    layers = list(backbone.children())[:-1]
    print(layers)


if __name__ == '__main__':
        ###########################################################################################
        # ARGS
        parser = argparse.ArgumentParser(description='Training')
        parser.add_argument('--loadchk', default='', help='Pass through to load training from a checkpoint')
        parser.add_argument('--datadir', default=settings3.output_directory, help='train directory')
        parser.add_argument('--gpu', default=0, type=int, help='GPU device number')
        parser.add_argument('--arch', default='alexnet', help='model architecture')
        parser.add_argument('--trainable_base', default=0, type=int, help='Whether to train the feature extractor')
        parser.add_argument('--rnn_model', default='LSTM', type=str, help='RNN model at clasification')
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

        parser.add_argument('--input_file', default=f'{homedir}/task1/models/two_stream/test/test1.mp4', help='Input file to predict')
        parser.add_argument('--checkpoint_path', default=f'/home/richard_bao/robosurgery/checkpoints/task1/two_stream/AC_CE_EF_FG/_ckpt_epoch_46.ckpt', help='path to load checkpoints')
        parser.add_argument('--hparams_path', default=f'{homedir}/task1/models/two_stream/test/hparams.yaml', help='path to load hyperparameters')

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
        #####################################################################################
        # Instantiate model
        torch.backends.cudnn.deterministic=True
        print("==> creating model FUSION '{}' '{}'".format(hparams.arch, hparams.rnn_model))
        model = TransferLearning(hparams)
        #####################################################################################
        print('Logging to: % s' % hparams.logging_dir)
        logger = TensorBoardLogger(hparams.logging_dir, name='%s/transfer_%s_%s_%s%s' %(classification_name, hparams.arch, hparams.trainable_base, hparams.rnn_model, hparams.masked))
        logger.log_hyperparams(hparams) # Log the hyperparameters
        # Set default device
        # torch.cuda.set_device(hparams.gpu)

        checkpoint_callback = ModelCheckpoint(
                # filepath=os.path.join(logger.log_dir, 'checkpoints'),
                filepath=os.path.join(settings3.checkpoints, classification_name, "transfer",  f'{hparams.arch}_{hparams.trainable_base}_{hparams.rnn_model}{hparams.masked}'),
                save_top_k=3,
                verbose=True,
                monitor='val_acc',
                mode='max',
                prefix='')

        kwargs = {'gpus': [hparams.gpu], 'logger':logger, 'check_val_every_n_epoch':1,
                                'accumulate_grad_batches':hparams.accum_batches, 'fast_dev_run' :False,
                                'num_sanity_val_steps':0, 'reload_dataloaders_every_epoch':False,
                                'max_epochs' : hparams.epochs, 'log_save_interval':200, 'profiler':False,
                                'gradient_clip_val':0, 'terminate_on_nan':True,  'auto_lr_find': hparams.auto_lr,
                                'track_grad_norm': 2, 'checkpoint_callback':checkpoint_callback} # overfit_pct =0.01

        if hparams.overfit == 1:
                kwargs['overfit_pct'] = 0.05


        if hparams.loadchk == '':
                trainer = Trainer(**kwargs)
        else:
                trainer = Trainer(resume_from_checkpoint = haparams.loadchk, **kwargs)
        trainer.fit(model)


