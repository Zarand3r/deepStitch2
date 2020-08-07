
import sys
sys.path.append('/home/fluongo/code/usc_project/FlowNetPytorch')

import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb

import cv2
import os
import glob
from PIL import Image

# DALI imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
import tempfile

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


def generate_temp_file(f, balance_classes = False):
    """Generates a temporary file from the directory information"""
    write_to_file = ""
    write_to_file += "%s %d\n" % (f, 0)
    tf = tempfile.NamedTemporaryFile()
    tf.write(str.encode(write_to_file))
    tf.flush()
    return tf

def flow2rgb_torch(input):
    """ Convert optical flow into an RGB coded tensor, uses pytorch and performs on GPU"""
    nB, nF, nH, nW, _ = input.size()

    rgb_optic_flow = torch.zeros(nB, nF, nH, nW, 3)
    for bb in range(nB):
        u = torch.clamp(input[bb, :, :, :, 0], -100, 100)/ 100. # Clip
        v = torch.clamp(input[bb, :, :, :, 1], -100, 100)/ 100. # Clip
        rgb_optic_flow[bb, :, :, :, 0] = u
        rgb_optic_flow[bb, :, :, :, 1] = 0.5 * (u + v)
        rgb_optic_flow[bb, :, :, :, 2] = v
    return torch.clamp(rgb_optic_flow, 0, 1)

class OFPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data, seq_len, step):
        super(OFPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.VideoReader(device="gpu", file_list=data, sequence_length=seq_len, 
                    step = step)
        self.of_op = ops.OpticalFlow(device="gpu", output_format=4)

    def define_graph(self):
        seq, labels = self.input(name="Reader")
        of = self.of_op(seq)
        return seq, of, labels # Return both the of and seq

##############################################################
parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--mp4_fn', default='',type=str, help='input path')
parser.add_argument('--gpu_id', default=1,type=int, help='which gpu')
args = parser.parse_args()

mp4_fn = args.mp4_fn
gpu_id = args.gpu_id

if __name__ == '__main__':
    print(mp4_fn)
    #mp4_fn = '/home/fluongo/code/usc_project/FlowNetPytorch/gesture13_stitch1285_cfr.mp4'
    fn_out = os.path.join('/'.join(mp4_fn.split('/')[:-1]), 'flowdali_'+ mp4_fn.split('/')[-1])
    torch.cuda.set_device(gpu_id) 

    tf = generate_temp_file(mp4_fn)

    mp4_ims = mp4_load(mp4_fn)

    load_full = True
    if load_full:
        pipe1 = OFPipeline(batch_size=1, num_threads=4, device_id=gpu_id, data = tf.name, seq_len=len(mp4_ims), step = 1)
    else:
        pipe1 = OFPipeline(batch_size=1, num_threads=4, device_id=gpu_id, data = tf.name, seq_len=3, step = 2)
    pipe1.build()
    dali_loader = DALIGenericIterator(pipe1, ["rgb", "optic_flow", "labels"], pipe1.epoch_size()['Reader'], 
                                        fill_last_batch = True, last_batch_padded = True)

    dali_loader.reset() # Reset the loader
    if load_full:
        for out in dali_loader:
            print('loaded %s' % mp4_fn)
    else:
        inputs_list_of = []; inputs_list_rgb = []
        for out in dali_loader:
            curr_rgb    = out[0]['rgb'][:, 1:, :, :, :].clone().detach()
            curr_of     = out[0]['optic_flow'][:, :, :, :, :].clone().detach()
            
            inputs_list_rgb.append(curr_rgb) # Middle frame
            inputs_list_of.append(curr_of) # Last OF frame of the 2

    if load_full:
        flow_out = 1-flow2rgb_torch(out[0]['optic_flow'].cpu())
    else:
        flow_out = 1-flow2rgb_torch(torch.cat(inputs_list_of, axis = 1).cpu())
    flow_out = F.interpolate(flow_out.permute(0, 1, 4, 2, 3).squeeze(), size=mp4_ims[0].shape[:2], mode='bilinear', align_corners=False)

    flow_rgb = (flow_out.cpu().detach().numpy()*255.).astype(np.uint8)

    [nY, nX, nC] = mp4_ims[0].shape

    export_ims = []
    for aa in range(flow_rgb.shape[0]):
        combined = np.zeros([nY, nX*2, nC], dtype=np.uint8)
        combined[:, :nX, :] = mp4_ims[aa][:, :, [2,1,0]] # Invert back for BGR
        combined[:, nX:, :] = flow_rgb[aa, :, :, :].transpose(1,2,0)
        export_ims.append(combined) # Save for list

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fn_out,fourcc, 30.0, (int(2*nX),int(nY)))

    for frame in export_ims:
        out.write(frame)
    out.release()

