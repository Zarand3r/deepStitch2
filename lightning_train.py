import os, argparse, glob, random, tempfile
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

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

class CustomDataset(Dataset):
	"""CustomDataset"""
	def __init__(self, global_dir, idxs= None, include_classes = [], flow_method = 'dali', balance_classes = False):
		self.global_dir = global_dir
		if len(include_classes) == 0:
			self.classes = os.listdir(global_dir)
			fns = glob.glob(os.path.join(global_dir, '*', 'flow%s*' % flow_method))
		else:
			self.classes = include_classes
			fns = []
			for class_curr in include_classes:
				fns.extend(glob.glob(os.path.join(global_dir, class_curr, 'flow%s*' % flow_method)))
		if len(fns) == 0:
			raise ValueError('Likely that you have not pre-computed the optical flow')
		if idxs == None: # load all
			idxs = list(range(len(fns)))
		self.filtered_fns = [[f, self.classes.index(f.split('/')[-2]) ] for i, f in enumerate(fns) if i in idxs]
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
		if n_frames > 300: # Chop off to last 1000
			video = video[-299:, :, :]
		start_phase = random.choice([0, 1, 2])
		video = video[list(range(start_phase, video.size()[0], 3)), :, :]
		return (video, label)

class FusionModel(LightningModule):
	def __init__(self, args):
		super(FusionModel, self).__init__()
		self.hparams = args
		self.actual = []; self.actual_train = []
		self.predicted = []; self.predicted_train = []
		self.batch_size = 1
		############################################################################################
		# Generate the train and test splits
		train_proportion = 0.8; fns = []
		for class_curr in self.hparams.include_classes:
			fns.extend(glob.glob(os.path.join(self.hparams.datadir, class_curr, 'flow%s*' % self.hparams.flow_method)))
		idx = list(range(len(fns)))
		random.seed(1); random.shuffle(idx)
		self.hparams.idx_train 	= idx[:int(train_proportion*len(idx))].copy() # Save as hyperparams
		self.hparams.idx_test 	= idx[int(train_proportion*len(idx)):].copy() # Save as hyperparams
		############################################################################################

		# Model specific
		original_model = models.__dict__[args.arch](pretrained=True)
		self.hidden_size = args.hidden_size
		self.num_classes = len(args.include_classes)
		self.fc_size = args.fc_size
		self.trainable_base = args.trainable_base

		# select a base model
		if args.arch.startswith('alexnet'):
			self.features = original_model.features
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = self.trainable_base
			# Make the output of each one be to fc_size/2 so that we cooncat the two fc outputs
			self.fc_pre = nn.Sequential(nn.Linear(9216, int(args.fc_size/2) ), nn.Dropout())
		elif args.arch.startswith('vgg16'):
			self.features = original_model.features
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = self.trainable_base
			# Make the output of each one be to fc_size/2 so that we cooncat the two fc outputs
			self.fc_pre = nn.Sequential(nn.Linear(25088, int(args.fc_size/2) ), nn.Dropout())
		elif args.arch.startswith('resnet18'):
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = self.trainable_base
			self.fc_pre = nn.Sequential(nn.Linear(512, int(args.fc_size/2)), nn.Dropout())
		elif args.arch.startswith('resnet34'):
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = self.trainable_base
			self.fc_pre = nn.Sequential(nn.Linear(512, int(args.fc_size/2)), nn.Dropout())
		elif args.arch.startswith('resnet50'):
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = self.trainable_base
			self.fc_pre = nn.Sequential(nn.Linear(2048, int(args.fc_size/2)), nn.Dropout())
		elif args.arch.startswith('resnet101'):
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = self.trainable_base
			self.fc_pre = nn.Sequential(nn.Linear(2048, int(args.fc_size/2)), nn.Dropout())
		else:
			raise ValueError('architecture base model not yet implemented choices: alexnet, ResNet 18/34/50')
		# Select an RNN
		if args.rnn_model == 'LSTM':
			self.rnn = nn.LSTM(input_size = args.fc_size,
						hidden_size = args.hidden_size,
						num_layers = args.rnn_layers,
						batch_first = True)
		elif args.rnn_model == 'RNN':
			self.rnn = nn.RNN(input_size = args.fc_size,
						hidden_size = args.hidden_size,
						num_layers = args.rnn_layers,
						batch_first = True)
		elif args.rnn_model == 'GRU':
			self.rnn = nn.RNN(input_size = args.fc_size,
						hidden_size = args.hidden_size,
						num_layers = args.rnn_layers,
						batch_first = True)
		else:
			raise ValueError('Not implemented, choose RNN/LSTM/GRU')
		
		self.fc = nn.Linear(args.hidden_size, self.num_classes)
		self.modelName = '%s_%s_latefusion_trainbaseparams_%s' % (args.arch, args.rnn_model, args.trainable_base)
	
	def init_hidden(self, num_layers, batch_size):
		return (torch.zeros(num_layers, batch_size, self.hidden_size).cuda(),
				torch.zeros(num_layers, batch_size, self.hidden_size).cuda())

	def forward(self, inputs, hidden=None, steps=0):
		nBatch, nFrames, ofH, ofW, nChannels, _ = inputs.shape
		fs = torch.zeros(nBatch, nFrames, self.rnn.input_size).cuda()

		for kk in range(nFrames):
			f_all = []
			f = self.features(inputs[:, kk, :, :, :, 0].permute(0, 3, 1, 2)) # permute to nB x nC x H x W
			f = f.reshape(f.size(0), -1)
			f = self.fc_pre(f)
			f_all.append(f)
			f_of = self.features(inputs[:, kk, :, :, :, 1].permute(0, 3, 1, 2))  # permute to nB x nC x H x W
			f_of = f_of.reshape(f_of.size(0), -1)
			f_of = self.fc_pre(f_of)
			f_all.append(f_of)

			# Concat
			f_cat = torch.cat(f_all, dim=-1)
			#fs[:, kk, :] = self.bn(f_cat)
			fs[:, kk, :] = f_cat

		# Note that for the 2 layer network will output hidden state as
		outputs, (hidden, cell)  = self.rnn(fs, hidden)
		outputs = self.fc(outputs)

		return outputs, hidden, cell

	def training_step(self, batch, batch_idx):
		# Batch is already on GPU by now
		input_cuda, target_cuda = self.apply_transforms_GPU(batch)
		output, _, _ = self(input_cuda)
		output = output[:, -1, :]
		loss = F.cross_entropy(output, target_cuda.type(torch.long))
		self.actual_train.append(target_cuda.item())
		self.predicted_train.append(output.topk(1,1)[-1].item())
		tensorboard_logs = {'train/loss': loss}
		return {'loss': loss, 'log': tensorboard_logs}

	def validation_step(self, batch, batch_idx):
		input_cuda, target_cuda = self.apply_transforms_GPU(batch)
		output, _, _ = self(input_cuda)
		output = output[:, -1, :]
		loss = F.cross_entropy(output, target_cuda.type(torch.long))
		self.actual.append(target_cuda.item())
		self.predicted.append(output.topk(1,1)[-1].item())
		return {'val_loss': loss}

	def validation_end(self, outputs):
		########### Log to tensorboard#######################################################
		top1_val = self.send_im_calculate_top1(self.actual, self.predicted, cmap_use = 'Blues', name = 'val/conf_mat')
		top1_train = self.send_im_calculate_top1(self.actual_train, self.predicted_train, cmap_use = 'Oranges', name = 'train/conf_mat')
		#####################################################################################
		self.actual = []; self.actual_train = []
		self.predicted = []; self.predicted_train = []
		
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		tensorboard_logs = {'val/loss': avg_loss, 'val/top1': top1_val, 'train/top1': top1_train, 'step': self.current_epoch}
		return {'val_loss': avg_loss, 'val_acc':top1_val, 'log': tensorboard_logs}
	
	def send_im_calculate_top1(self, actual, predicted, cmap_use = 'Blues', name = 'tmp/name'):
		cm = confusion_matrix(actual, predicted)
		fig = plt.figure(); sns.heatmap(cm, cmap = cmap_use, ax =plt.gca(), annot = True, xticklabels = self.hparams.include_classes, yticklabels = self.hparams.include_classes)
		self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch, close = True)
		top1 = float(sum([a==b for a,b in zip(actual, predicted)]))/len(actual)
		return top1
		
	def configure_optimizers(self):
		if self.trainable_base:
			optimizer = torch.optim.Adam([{'params': self.features.parameters()},
									{'params': self.fc_pre.parameters()},
									{'params': self.rnn.parameters()},
									{'params': self.fc.parameters()}],
									lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay = self.hparams.weight_decay)
		else:
			optimizer = torch.optim.Adam([{'params': self.fc_pre.parameters()},
									{'params': self.rnn.parameters()},
									{'params': self.fc.parameters()}],
									lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay = self.hparams.weight_decay)
		return optimizer

	def train_dataloader(self):
		train_dataset 	= CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_train , include_classes = self.hparams.include_classes, flow_method = self.hparams.flow_method, balance_classes=True)
		train_dataloader 	= DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
		self.epoch_len = len(train_dataset)
		return train_dataloader

	def val_dataloader(self):
		val_dataset 	= CustomDataset(self.hparams.datadir, idxs = self.hparams.idx_test , include_classes = self.hparams.include_classes, flow_method = self.hparams.flow_method)
		val_dataloader 	= DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
		return val_dataloader

	def apply_transforms_GPU(self, batch):
		# Prepares the outputs
		nB, nF, nH, nW, nC = batch[0].size()
		rgb = self.augGPU_resize(batch[0][:, :, :, :int(nW/2), :].type(torch.float)/255., npix_resize = (224, 224))
		#rgb = self.augGPU_normalize_inplace(rgb, mean = [0.3, 0.2, 0.2], std = [0.2, 0.2, 0.2])
		rgb = self.augGPU_normalize_inplace(rgb, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
		
		of 	= self.augGPU_resize(batch[0][:, :, :, int(nW/2):, :].type(torch.float)/255., npix_resize = (224, 224))
		#of = self.augGPU_normalize_inplace(of, mean = [0.01, 0.01, 0.01], std = [0.05, 0.05, 0.05])
		of = self.augGPU_normalize_inplace(of, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

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
	parser.add_argument('--datadir', default='/home/fluongo/code/usc_project/usc_data/balint/training_ready/cfr_cut_mov', help='train directory')
	parser.add_argument('--gpu', default=1, type=int, help='GPU device number')
	parser.add_argument('--arch', default='alexnet', help='model architecture')
	parser.add_argument('--trainable_base', default=0, type=int, help='Whether to train the feature extractor')
	parser.add_argument('--rnn_model', default='LSTM', type=str, help='RNN model at clasification')
	parser.add_argument('--rnn_layers', default=2, type=int, help='number of rnn layers')
	parser.add_argument('--hidden_size', default=16, type=int, help='output size of rnn hidden layers')
	parser.add_argument('--fc_size', default=32, type=int, help='size of fully connected layer before rnn')
	parser.add_argument('--epochs', default=30, type=int, help='manual epoch number')
	parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
	parser.add_argument('--lr_lambdas', default=0.9, type=float, help='Schedulre hyperparam')
	parser.add_argument('--include_classes', default='', type=str, help='Which classnames to include')
	parser.add_argument('--flow_method', default='flownet', type=str, help='Which flow method to use (flownet or dali)')
	parser.add_argument('--random_crop', default=0, type=int, help='Whether or not to augment with random crops...')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--weight_decay', default=0, type=float)
	parser.add_argument('--accum_batches', default=1, type=int)
	
	
	
	hparams = parser.parse_args()
	if hparams.trainable_base == 1:
		hparams.trainable_base = True
	else:
		hparams.trainable_base = False

	random_crop = False if hparams.random_crop == 1 else True
	if hparams.include_classes == '':
		hparams.include_classes = ['01', '02', '03', '07', '13']
	else:
		hparams.include_classes = hparams.include_classes.split(' ')
	#####################################################################################
	# Instantiate model
	print("==> creating model FUSION '{}' ".format(hparams.arch))
	model = FusionModel(hparams)
	#####################################################################################
	logger = TensorBoardLogger("lightning_logs", name='%s/%s_%s_%s' %(hparams.datadir.split('/')[-1], hparams.arch, hparams.trainable_base, hparams.rnn_model))
	logger.log_hyperparams(hparams)
	# Set default device
	# torch.cuda.set_device(hparams.gpu)

	# checkpoint_callback = ModelCheckpoint(
    # filepath='/path/to/store/weights.ckpt',
    # save_best_only=True,
    # verbose=True,
    # monitor='val_loss',
    # mode='min')


	kwargs = {'gpus': [hparams.gpu], 'logger':logger, 'check_val_every_n_epoch':1, 
				'accumulate_grad_batches':hparams.accum_batches, 'fast_dev_run' :False, 
				'num_sanity_val_steps':0, 'reload_dataloaders_every_epoch':False, 
				'max_epochs' : hparams.epochs, 'log_save_interval':200, 'profiler':False, 
				'gradient_clip_val':0, 'terminate_on_nan':True,  
				'track_grad_norm': 2}# overfit_pct =0.01
	if hparams.loadchk == '':
		# trainer = Trainer(gpus = [hparams.gpu], logger = logger, check_val_every_n_epoch=10, accumulate_grad_batches=1, fast_dev_run = False, 
		# 			num_sanity_val_steps=0, reload_dataloaders_every_epoch=False, 
		# 			max_epochs = hparams.epochs, log_save_interval=200, profiler=True, gradient_clip_val=0, terminate_on_nan = True,  
		# 			track_grad_norm = 2, overfit_pct = 0.05) # overfit_pct =0.01
		trainer = Trainer(**kwargs)
	else:
		trainer = Trainer(resume_from_checkpoint = haparams.loadchk, **kwargs)
	trainer.fit(model)
