# Temporally recurrent models of stitch recognition

## Overview





The goal of this project is to apply computer vision techniques (standard two-stream action recogntiion models) towards the goal of

1. Stitch type recognition according to the following defined taxonomy (courtesy of USC neuro-robotic group)

<p align="center">
  <img width="400" src=assets/gesture_types.png>
</p>

2. Recognition of when a stitch is happening (e.g. "Stitch segmentation")

Here is the general model schematic for these approaches in which we will vary the pre-trained backbone (e.g. alexNet, resnet, or VGG16) and the recurrent model learned at the classification layer (RNN, GRU, LSTM, convLSTM, convttLSTM). 




<p align="center">
  <img width="400" src=assets/model_schematic.png>
</p>

<p align="center">
  <img width="400" src=assets/example_ims.png>
</p>

## Prerequisites

PyTorch (1.3)
PyTorch lightning
ffmpeg 
NVIDIA DALI

Computation using flownet refers to the following github implementation of flownet: 
https://github.com/ClementPinard/FlowNetPytorch

If computing optical flow in a faster way you can use the NVIDIA DALI (https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) framework which uses the turingOF library from nvidia. This is substantially faster inference than flownet but is also coarser in its estimates resulting in slightly lower performance.



## Training

If you have the mp4 files in a labeled directory with labeled subdirs corresponding to gesture labels

```
-───movie_clips
│   └───00
│       │   fn*.mp4
|       |   ...
│   └───01
│       │   fn*.mp4
|       |   ...
```

```
# Preprocess the data into the corresponding movie folder
python preprocessing/generate_flows.sh movie_clips/

# Train the model
python lightning_train.py \
        --arch resnet18 \
        --rnn_model convLSTM \
        --rnn_layers 2 \
        --hidden_size 64 \
        --fc_size 128 \
        --lr 1e-4
```

*Instructions for cutting model to come later...

## Dataset (to be made available soon)

411 labeled video clips of varying length (2-30s) across the 5 gesture types shown above

24 surgical preocedure videos (>1hour) with labeled segments corresponding to times that the needle actually entered tissue

## Results

Models for recognizing and evaluating stitches in srgical videos using deep networks

Currently migrating over from old directory, this will be the newest one as it contains lightning files for expedited training during experiments.


<p align="left">
  <img width="200" src=assets/alexnet_convLSTM_confusionMat.png>
</p>