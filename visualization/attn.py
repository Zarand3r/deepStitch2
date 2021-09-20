import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import matplotlib.cm as cm
import os
#import imageio
from PIL import Image as im
#import cv2
smooth = True
#print(os.listdir())

filepath = "./attn.npy"
#attn shape (111, 1, 1, 6, 6) nFrames, nB, nC, H, W
attn = np.load(filepath)
#print(attn.shape)
# print()
attn = attn[:, 0, :, :, :]
attn = np.moveaxis(attn, 1, -1)
attn = attn[80, :, :, :]

#
filepath1 = './input.npy'
#video shape (1, 112, 224, 224, 3) nB, nFrames, H, W, Channels
video = np.load(filepath1)
print(video.shape)
image = video[0, 80, :, :, :]
#print(image.shape)
#image_data = im.fromarray(image)

#imageio.mimwrite('/home/idris/robosurgery/deepStitch2/task4/models/saved_vals/attn.mp4', val , fps = 1)



#image = Image.open(image_path)
#image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)


for t in range(1):
    plt.imshow(image)
    if smooth:
        alpha = skimage.transform.pyramid_expand(attn, upscale=3, sigma=6)
    if t == 0:
        plt.imshow(alpha, alpha=0.5)
    else:
        plt.imshow(alpha, alpha=0.8)
    plt.set_cmap("Reds")
    plt.axis('off')
plt.show()