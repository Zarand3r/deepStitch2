import sys
sys.path.append('/home/fluongo/code/usc_project/FlowNetPytorch')
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
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
parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--mp4_fn', default='',type=str, help='input path')
parser.add_argument('--gpu_id', default=1,type=int, help='which gpu')
args = parser.parse_args()

model_dir = '/home/fluongo/code/usc_project/FlowNetPytorch/flownets_EPE1.951.pth.tar'
mp4_fn = args.mp4_fn
gpu_id = args.gpu_id
torch.cuda.set_device(gpu_id) 



if __name__ == '__main__':
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    # create model
    network_data = torch.load(model_dir, map_location='cpu')
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).cuda()
    model.eval()
    cudnn.benchmark = True

    # if 'div_flow' in network_data.keys():
    #     args.div_flow = network_data['div_flow']


    mp4_ims = mp4_load(mp4_fn)
    n_im = len(mp4_ims) # Number of images

    export_ims = []
    [nY, nX, nC] = mp4_ims[0].shape

    #frame_ints = list(range(n_im))
    div_flow = 2
    max_flow = 20
    outs = []

    for ii in tqdm(range(n_im-1)):
        
        aa=ii
        bb=ii+1

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

        outs.append(np.array(output.cpu().detach()))

        # Upsample
        output = F.interpolate(output, size=img1.size()[-2:], mode='bilinear', align_corners=False)

        # Convert to an RGBim1.
        for suffix, flow_output in zip(['flow', 'inv_flow'], output):
            rgb_flow = flow2rgb(div_flow * flow_output, max_value=max_flow)
            to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)

        combined = np.zeros([nY, nX*2, nC], dtype=np.uint8)
        combined[:, :nX, :] = mp4_ims[aa][:, :, [2,1,0]] # Invert back for BGR
        combined[:, nX:, :] = to_save
        export_ims.append(combined) # Save for list

    # Write out the MP4

    fn_out = os.path.join('/'.join(mp4_fn.split('/')[:-1]), 'flowflownet_'+ mp4_fn.split('/')[-1])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fn_out,fourcc, 30.0, (int(2*nX),int(nY)))

    for frame in export_ims:
        out.write(frame)

    out.release()