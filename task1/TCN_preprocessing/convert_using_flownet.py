import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/utils')
import settings1
sys.path.append(settings1.flownet_dir) #This should include the modules from https://github.com/ClementPinard/FlowNetPytorch so import models works
import models

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb

import cv2
import os
import glob
from PIL import Image


def mp4_load(fn):
    # Take an MP4 video and return a list of frames
    vidcap = cv2.VideoCapture(fn)
    success,image = vidcap.read()
    out = []
    while True:
        success,image = vidcap.read()
        if not success:
            break
        else:
            image = np.array(Image.fromarray(image).resize((960, 540)))
            out.append(image[:, :, [2,1,0]])

    return out

##############################################################
 

class OpticalFlow:
    def __init__(self, args):
        self.args = args
        self.model_dir = settings1.flownet_model
        torch.cuda.set_device(self.args.gpu_id)

    def save_flow(self, flow_output, fname, index):
        for suffix, flow_output in zip(['flow', 'inv_flow'], flow_output):
            fdir= os.path.join(self.args.output_dir, fname)
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            fpath = os.path.join(fdir, f"{suffix}_{index}")
            if self.args.output_type in ['vis']:
                rgb_flow = flow2rgb(self.args.div_flow * flow_output, max_value=self.args.max_flow)
                to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
                # imwrite(filename + '.png', to_save)
            elif self.args.output_type in ['raw']:
                # Make the flow map a HxWx2 array as in .flo files
                to_save = (args.div_flow*flow_output).cpu().numpy().transpose(1,2,0)
                # np.save(filename + '.npy', to_save)
            elif self.args.output_type in ['gray']:
                to_save = (args.div_flow*flow_output).cpu().detach().numpy().transpose(1,2,0)
                print(to_save.shape)
                x_flow = to_save[:, :, 0]
                x_flow = np.expand_dims(x_flow, axis=-1)
                y_flow = to_save[:, :, 1]
                print(x_flow.shape)
                # split this 2 channel image into 2 grayscale (1 channel) images for x and y
                imwrite(fpath + '.png', x_flow)

            # make join an option for output_type

    def generate_flow(self):
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
        ])

        # create model
        network_data = torch.load(self.model_dir, map_location='cpu')
        print("=> using pre-trained model '{}'".format(network_data['arch']))
        model = models.__dict__[network_data['arch']](network_data).cuda()
        model.eval()
        cudnn.benchmark = True

        # if 'div_flow' in network_data.keys():
        #     args.div_flow = network_data['div_flow']

        mp4_list = glob.glob(os.path.join(self.args.input_dir, "*.m4v"))
        print(mp4_list)
        for mp4_fn in mp4_list:
            print(mp4_fn)
            mp4_ims = mp4_load(mp4_fn)
            n_im = len(mp4_ims) # Number of images

            export_ims = []
            [nY, nX, nC] = mp4_ims[0].shape

            #frame_ints = list(range(n_im))

            for ii in tqdm(range(n_im-self.args.window)):
                if ii > 10:
                    break
                aa=ii
                bb=ii+self.args.window
                img1 = input_transform(mp4_ims[aa])
                img2 = input_transform(mp4_ims[bb])
                input_var = torch.cat([img1, img2]).unsqueeze(0)
                # if args.bidirectional:
                # feed inverted pair along with normal pair
                inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
                input_var = torch.cat([input_var, inverted_input_var])
                input_var = input_var.cuda()
                # compute output
                output = model(input_var)
                # outs.append(np.array(output.cpu().detach()))
                # Upsample
                output = F.interpolate(output, size=img1.size()[-2:], mode='bilinear', align_corners=False)
                fname = os.path.basename(mp4_fn).split(".")[0]
                self.save_flow(output, fname, ii)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conversion')
    parser.add_argument('--input_dir', default=settings1.TCN_raw,type=str, help='directory with the mp4 videos')
    parser.add_argument('--output_dir', default=settings1.TCN_data,type=str, help='directory to output the optical flows')
    parser.add_argument('--gpu_id', default=1,type=int, help='which gpu')
    parser.add_argument('--window', default=4,type=int, help='sliding window for smoothing frames')
    parser.add_argument('--div_flow', default=20, type=float, help='value by which flow will be divided. overwritten if stored in pretrained file')
    parser.add_argument('--max_flow', default=None, type=float, help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
    parser.add_argument('--output_type', default='gray', help='viz or raw or gray')
    args = parser.parse_args()

    Flow1 = OpticalFlow(args)
    Flow1.generate_flow()
