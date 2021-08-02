import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
from lightning_train1 import LightningModule
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import cv2
#from scipy import imread, imresize
from PIL import Image


def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("could not open: ", video_path)
        return -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    return length, width, height


def visualize_att(image_path, alphas, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    model = LightningModule()
    model.load_state_dict(torch.load(alphas))
    print(model)



    length, _, _ = video_frame_count(image_path)
    cap = cv2.VideoCapture(image_path)
    # q = queue.Queue(self.frames_num)
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
        current_alpha = alphas
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=36, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
    plt.imshow(alpha, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--img_path', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')

    args = parser.parse_args()

    visualize_att(args.img_path, args.model)