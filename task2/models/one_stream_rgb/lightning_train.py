import os, argparse, glob, random, tempfile
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.special import softmax
import numpy as np

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
from convlstmcells import ConvLSTMCell, ConvTTLSTMCell
import settings

class CustomDataset(Dataset):
	"""CustomDataset"""
	def __init__(self, global_dir, idxs= None, include_classes = [], flow_method = 'flownet', balance_classes = False, mode = 'train', max_frames = 150, stride = 2, masked=""):
		self.global_dir = global_dir
		self.mode = mode
		self.max_frames = stride*max_frames
		self.stride = stride
		if len(include_classes) == 0:
			self.classes = os.listdir(global_dir)
			fns = glob.glob(os.path.join(global_dir, '*', '%s*' % flow_method))
		else:
			self.classes = include_classes
			fns = []
			for class_curr in include_classes:
				fns.extend(glob.glob(os.path.join(global_dir, class_curr, "optical_flow"+masked, '%s*' % flow_method)))
		if len(fns) == 0:
			raise ValueError('Likely that you have not pre-computed the optical flow or data directory is wrong!')
		if idxs == None: # load all
			idxs = list(range(len(fns)))
		self.filtered_fns = [[f, self.classes.index(f.split('/')[-3]) ] for i, f in enumerate(fns) if i in idxs]
		if balance_classes:
			class_counter = Counter([f[1] for f in self.filtered_fns])
			print(class_counter)
			print('balancing...')
			n_match = class_counter.most_common()[0][1]
			for class_curr in class_counter.keys():
				cnt = class_counter[class_curr]
				print(class_curr, cnt)
				self.filtered_fns.extend(random.choices([f for f in self.filtered_fns if f[1] == int(class_curr)], k=max(n_match-cnt, 1) ))
			print(Counter([f[1] for f in self.filtered_fns]))
			print('Classes now balanced')

	def __len__(self):
		return len(self.filtered_fns)

	def __getitem__(self, idx):
		if torchvision.__version__[:3] == '0.4':
			video = torchvision.io.read_video(self.filtered_fns[idx][0])[0]
		else: # Newer version
			video = torchvision.io.read_video(self.filtered_fns[idx][0], pts_unit = 'sec')[0]
		label = self.filtered_fns[idx][1]
		n_frames = video.size()[0]
		if self.mode == 'train':
			if n_frames > self.max_frames: # Sample random 200 frames
				start_ii = random.choice(list(range(0, n_frames-self.max_frames)))
				video = video[start_ii:start_ii+(self.max_frames-1), :, :]
			start_phase = random.choice(list(range(self.stride)))
			video = video[list(range(start_phase, video.size()[0], self.stride)), :, :]
		elif self.mode == 'val':
			if n_frames > self.max_frames: # Sample random 200 frames
				start_ii = random.choice(list(range(0, n_frames-self.max_frames)))
				video = video[start_ii:start_ii+(self.max_frames-1), :, :]
			start_phase = random.choice(list(range(self.stride)))
			video = video[list(range(start_phase, video.size()[0], self.stride)), :, :]
		else:
			raise ValueError('not supported mode must be train or test')
		return (video, label)

class FusionModel(LightningModule):
	def __init__(self, args):
		super(FusionModel, self).__init__()
		self.hparams = args
		self.actual = []; self.actual_train = []
		self.predicted = []; self.predicted_train = []
		self.predicted_softmax = []; self.predicted_softmax_train = []
		self.batch_size = self.hparams.batch_size
		############################################################################################
		# Generate the tra300in and test splits
		fns = []
		for class_curr in self.hparams.include_classes:
			fns.extend(glob.glob(os.path.join(self.hparams.datadir, class_curr, "optical_flow"+self.hparams.masked, '%s*' % self.hparams.flow_method)))
		idx = list(range(len(fns)))
		random.seed(self.hparams.seed); random.shuffle(idx)
		self.hparams.idx_train 	= idx[:int(self.hparams.train_proportion*len(idx))].copy() # Save as hyperparams
		self.hparams.idx_test 	= idx[int(self.hparams.train_proportion*len(idx)):].copy() # Save as hyperparams
		############################################################################################

		# Model specific
		original_model_rgb = models.__dict__[args.arch](pretrained=self.hparams.use_pretrained)
		# original_model_of = models.__dict__[args.arch](pretrained=self.hparams.use_pretrained)
		
		self.hidden_size = args.hidden_size
		self.num_classes = len(args.include_classes)
		self.fc_size = args.fc_size
		self.trainable_base = args.trainable_base

		# select a base model
		if args.arch.startswith('alexnet'):
			self.features_rgb = original_model_rgb.features
			for i, param in enumerate(self.features_rgb.parameters()):
				param.requires_grad = self.trainable_base
			# self.features_of = original_model_of.features
			# for i, param in enumerate(self.features_of.parameters()):
			# 	param.requires_grad = self.trainable_base
			
			#Make the output of each one be to fc_size/2 so that we cooncat the two fc outputs
			self.fc_pre_rgb = nn.Sequential(nn.Linear(256*6*6, int(args.fc_size/2) ), nn.Dropout()) if 'conv' not in args.rnn_model else None
			# self.fc_pre_of = nn.Sequential(nn.Linear(256*6*6, int(args.fc_size/2) ), nn.Dropout()) if 'conv' not in args.rnn_model else None
			self.final_channels = 256

		elif args.arch.startswith('resnet18'):
			# self.features = nn.Sequential(*list(original_model.children())[:-2])
			# for i, param in enumerate(self.features.parameters()):
			# 	param.requires_grad = self.trainable_base
			# self.fc_pre = nn.Sequential(nn.Linear(512*7*7, int(args.fc_size/2)), nn.Dropout()) if 'conv' not in args.rnn_model else None
			# self.final_channels = 512
			self.features_rgb = nn.Sequential(*list(original_model_rgb.children())[:-2])
			for i, param in enumerate(self.features_rgb.parameters()):
				param.requires_grad = self.trainable_base
			# self.features_of = nn.Sequential(*list(original_model_of.children())[:-2])
			# for i, param in enumerate(self.features_of.parameters()):
			# 	param.requires_grad = self.trainable_base
			
			self.fc_pre_rgb = nn.Sequential(nn.Linear(512*7*7, int(args.fc_size/2) ), nn.Dropout()) if 'conv' not in args.rnn_model else None
			# self.fc_pre_of = nn.Sequential(nn.Linear(512*7*7, int(args.fc_size/2) ), nn.Dropout()) if 'conv' not in args.rnn_model else None
			self.final_channels = 512

		else:
			raise ValueError('architecture base model not yet implemented choices: alexnet, ResNet 18')
		# Select an RNN
		#print(self.features)
		# if args.rnn_model == 'LSTM':
		# 	self.rnn = nn.LSTM(input_size = int(args.fc_size/2),
		# 				hidden_size = args.hidden_size,
		# 				num_layers = args.rnn_layers,
		# 				batch_first = True)
		# 	self.fc = nn.Linear(args.hidden_size, self.num_classes)
		# elif args.rnn_model == 'convLSTM': 
		# 	# Twice number of channels for RGB and OF which are concat
		# 	self.rnn = ConvLSTMCell(input_channels = self.final_channels, hidden_channels = self.final_channels, kernel_size = 3, bias = True)
			
		# 	nF = 6 if args.arch.startswith('alexnet') else 7
		# 	self.fc = nn.Linear(self.final_channels*nF*nF, self.num_classes)
		# elif args.rnn_model == 'convttLSTM': 
		# 	# Twice number of channels for RGB and OF which are concat
		# 	self.rnn = ConvTTLSTMCell(input_channels = self.final_channels, hidden_channels = int(self.final_channels/2), order = 3, steps = 5, ranks = 16, kernel_size = 3, bias = True)
			
		# 	nF = 6 if args.arch.startswith('alexnet') else 7
		# 	self.fc = nn.Linear(self.final_channels*nF*nF, self.num_classes)
		# else:
		# 	raise ValueError('Not implemented, choose LSTM, convLSTM, convttLSTM type')
		
		# self.modelName = '%s_%s_latefusion_trainbaseparams_%s' % (args.arch, args.rnn_model, args.trainable_base)
		if args.rnn_model == 'LSTM':
			self.rnn = nn.LSTM(input_size = int(args.fc_size/2),
						hidden_size = args.hidden_size,
						num_layers = args.rnn_layers,
						batch_first = True)
			self.fc = nn.Linear(args.hidden_size, self.num_classes)
		elif args.rnn_model == 'convLSTM': 
			# Twice number of channels for RGB and OF which are concat
			self.rnn = ConvLSTMCell(input_channels = self.final_channels, hidden_channels = int(self.hparams.hidden_size), kernel_size = 3, bias = True)
			
			nF = 6 if args.arch.startswith('alexnet') else 7
			self.fc = nn.Linear(int(self.hparams.hidden_size)*nF*nF, self.num_classes) #replace self.final_channels here the parameter must equal the hidden_channels in self.rnn
		elif args.rnn_model == 'convttLSTM': 
			# Twice number of channels for RGB and OF which are concat
			self.rnn = ConvTTLSTMCell(input_channels = self.final_channels, hidden_channels = int(self.final_channels/2), order = 3, steps = 5, ranks = 16, kernel_size = 3, bias = True)
			
			nF = 6 if args.arch.startswith('alexnet') else 7
			self.fc = nn.Linear(self.final_channels*nF*nF, self.num_classes)
		else:
			raise ValueError('Not implemented, choose LSTM, convLSTM, convttLSTM type')
		
		self.modelName = '%s_%s_latefusion_trainbaseparams_%s' % (args.arch, args.rnn_model, args.trainable_base)
	
	def init_hidden(self, num_layers, batch_size):
		return (torch.zeros(num_layers, batch_size, self.hidden_size).cuda(),
				torch.zeros(num_layers, batch_size, self.hidden_size).cuda())

	def forward(self, inputs, hidden=None, steps=0):
		nBatch, nFrames, ofH, ofW, nChannels, _ = inputs.shape
		
		#########################################################################################
		# Non convolutional (old way)
		if 'conv' not in self.hparams.rnn_model:
			fs = torch.zeros(nBatch, nFrames, self.rnn.input_size).cuda()
			for kk in range(nFrames):
				f_all = []
				f = self.features_rgb(inputs[:, kk, :, :, :, 0].permute(0, 3, 1, 2)) # permute to nB x nC x H x W
				f = f.reshape(f.size(0), -1)
				f = self.fc_pre_rgb(f)
				f_all.append(f)

				# f_of = self.features_of(inputs[:, kk, :, :, :, 1].permute(0, 3, 1, 2))  # permute to nB x nC x H x W
				# f_of = f_of.reshape(f_of.size(0), -1)
				# f_of = self.fc_pre_of(f_of)
				# f_all.append(f_of)

				# Concat
				f_cat = torch.cat(f_all, dim=-1)
				fs[:, kk, :] = f_cat

			# Note that for the 2 layer network will output hidden state as
			outputs, (hidden, cell)  = self.rnn(fs, hidden)
			outputs = self.fc(outputs)
		else:
			#########################################################################################
			# Convolutional flavors
			for kk in range(nFrames):
				f_rgb = self.features_rgb(inputs[:, kk, :, :, :, 0].permute(0, 3, 1, 2)) # permute to nB x nC x H x W
				# f_of = self.features_of(inputs[:, kk, :, :, :, 1].permute(0, 3, 1, 2))  # permute to nB x nC x H x W
				
				# # Size nBatch x nChannels x H x W
				# if kk == 0:
				# 	outputs = self.rnn(torch.cat([f, f_of], dim = 1), first_step=True)
				# else:
				# 	outputs = self.rnn(torch.cat([f, f_of], dim = 1), first_step=False)
				# Size nBatch x nChannels x H x W
				if kk == 0:
					outputs = self.rnn(f_rgb, first_step=True)
				else:
					outputs = self.rnn(f_rgb, first_step=False)
			outputs = outputs.reshape(outputs.size(0), -1)
			outputs = self.fc(outputs)
			# Add a dimension to make the size consistent with old rnn
			outputs = outputs.unsqueeze(1)

			hidden, cell = _, _ # Implement how to do this later...
			#########################################################################################
		
		return outputs, hidden, cell

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
		self.predicted_softmax.append(softmax(output.detach().cpu().numpy(), axis = -1)) # Save scores

		return {'val_loss': loss}

	def validation_end(self, outputs):
		########### Log to tensorboard#######################################################
		if self.trainer.overfit_pct==0: # Not a debug loop, otherwise roc_auc can throw errors
			if self.num_classes ==2:
				auc_train 	= roc_auc_score(self.actual_train, np.vstack(self.predicted_softmax_train)[:, 1])
				auc_val 	= roc_auc_score(self.actual, np.vstack(self.predicted_softmax)[:, 1])
			else:
				auc_train 	= roc_auc_score(self.actual_train, np.vstack(self.predicted_softmax_train), multi_class = 'ovr')
				auc_val 	= roc_auc_score(self.actual, np.vstack(self.predicted_softmax), multi_class = 'ovr')
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
		cm = confusion_matrix(actual, predicted)
		fig = plt.figure(); sns.heatmap(cm, cmap = cmap_use, ax =plt.gca(), annot = True, xticklabels = self.hparams.include_classes, yticklabels = self.hparams.include_classes)
		self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch, close = True)
		top1 = float(sum([a==b for a,b in zip(actual, predicted)]))/len(actual)
		return top1
		
	def configure_optimizers(self):
		self.layers_to_fit = [{'params': self.fc.parameters()}, {'params': self.rnn.parameters()}]
		if self.fc_pre_rgb != None:
			self.layers_to_fit.append({'params': self.fc_pre_rgb.parameters()})
			# self.layers_to_fit.append({'params': self.fc_pre_of.parameters()})
		if self.trainable_base:
			self.layers_to_fit.append({'params': self.features_rgb.parameters()})
			# self.layers_to_fit.append({'params': self.features_of.parameters()})
		
		optimizer = torch.optim.Adam(self.layers_to_fit,
								lr=self.hparams.lr, betas=(0.9, 0.999), 
								weight_decay = self.hparams.weight_decay)
	
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.1)
		return [optimizer], [scheduler]

	def train_dataloader(self):
		train_dataset 	= CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_train , include_classes = self.hparams.include_classes, 
							flow_method = self.hparams.flow_method, balance_classes=True, mode = 'train', max_frames = self.hparams.loader_nframes,
							stride = self.hparams.loader_stride, masked = self.hparams.masked)
		train_dataloader 	= DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.hparams.number_workers, drop_last=True)
		self.epoch_len = len(train_dataset)
		return train_dataloader

	def val_dataloader(self):
		val_dataset 	= CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_test , include_classes = self.hparams.include_classes, 
							flow_method = self.hparams.flow_method, balance_classes=False, mode = 'val', max_frames = self.hparams.loader_nframes,
							stride = self.hparams.loader_stride, masked = self.hparams.masked)
		val_dataloader 	= DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.hparams.number_workers,drop_last=True)
		return val_dataloader

	def apply_transforms_GPU(self, batch, random_crop = False):
		# Prepares the outputs
		nB, nF, nH, nW, nC = batch[0].size()
		if random_crop:
			rgb = self.augGPU_resize(batch[0][:, :, :, :int(nW/2), :].type(torch.float)/255., npix_resize = (240, 240), random_crop = True)
			of 	= self.augGPU_resize(batch[0][:, :, :, int(nW/2):, :].type(torch.float)/255., npix_resize = (240, 240), random_crop = True)
		else:
			rgb = self.augGPU_resize(batch[0][:, :, :, :int(nW/2), :].type(torch.float)/255., npix_resize = (224, 224))
			of 	= self.augGPU_resize(batch[0][:, :, :, int(nW/2):, :].type(torch.float)/255., npix_resize = (224, 224))
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

if __name__ == '__main__':
	###########################################################################################
	# ARGS
	parser = argparse.ArgumentParser(description='Training')
	parser.add_argument('--loadchk', default='', help='Pass through to load training from a checkpoint')
	parser.add_argument('--datadir', default=settings.data_directory, help='train directory')
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
	parser.add_argument('--accum_batches', default=1, type=int)
	parser.add_argument('--overfit', default=0, type=int)
	parser.add_argument('--auto_lr', default=0, type=int)
	parser.add_argument('--use_pretrained', default=1, type=int, help='whether or not to load pretrained weights')
	parser.add_argument('--logging_dir', default='lightning_logs', type=str)
	parser.add_argument('--loader_nframes', default=140, type=int, help='How many frames to load at stride 2') #change to 140?
	parser.add_argument('--loader_stride', default=2, type=int, help='stride for dataloader')
	parser.add_argument('--number_workers', default=0, type=int, help='number of workers for Dataloader')
	parser.add_argument('--batch_size', default=1, type=int, help='batch size')
	parser.add_argument('--masked', dest='masked', action='store_true', help = "train on masked?")
	
	hparams = parser.parse_args()
	hparams.trainable_base = True if hparams.trainable_base == 1 else False
	hparams.random_crop = True if hparams.random_crop == 1 else False
	hparams.auto_lr = True if hparams.auto_lr == 1 else False
	hparams.use_pretrained = True if hparams.use_pretrained == 1 else False
	scale_batch = False
	if hparams.masked:
		hparams.masked = "_masked"
	else:
		hparams.masked = ""
	if hparams.batch_size < 1:
		scale_batch='power'

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
	model = FusionModel(hparams)
	#####################################################################################
	print('Logging to: % s' % hparams.logging_dir)
	logger = TensorBoardLogger(hparams.logging_dir, name='%s/%s_%s_%s%s' %(classification_name, hparams.arch, hparams.trainable_base, hparams.rnn_model, hparams.masked))
	logger.log_hyperparams(hparams) # Log the hyperparameters
	# Set default device
	# torch.cuda.set_device(hparams.gpu)

	checkpoint_callback = ModelCheckpoint(
		# filepath=os.path.join(logger.log_dir, 'checkpoints'),
		filepath=os.path.join(settings.checkpoints2, "one_stream_rgb", classification_name, f'{hparams.arch}_{hparams.trainable_base}_{hparams.rnn_model}{hparams.masked}'),
		save_top_k=3,
		verbose=True,
		monitor='val_acc',
		mode='max',
		prefix='')

	kwargs = {'gpus': [hparams.gpu], 'logger':logger, 'check_val_every_n_epoch':1, 
				'accumulate_grad_batches':hparams.accum_batches, 'fast_dev_run' :False, 
				'num_sanity_val_steps':0, 'reload_dataloaders_every_epoch':False, 
				'max_epochs' : hparams.epochs, 'log_save_interval':200, 'profiler':False, 
				'gradient_clip_val':0, 'terminate_on_nan':True, 'auto_lr_find': hparams.auto_lr, 
				'track_grad_norm':2, 'checkpoint_callback':checkpoint_callback, 'auto_scale_batch_size':scale_batch} # overfit_pct =0.01
	
	if hparams.overfit == 1:
		kwargs['overfit_pct'] = 0.05

	
	if hparams.loadchk == '':
		trainer = Trainer(**kwargs)
	else:
		trainer = Trainer(resume_from_checkpoint = haparams.loadchk, **kwargs)
	trainer.fit(model)

	# Gradient clipping, acum_batches, drop_last=true, and batches

