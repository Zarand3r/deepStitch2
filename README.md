# Temporally recurrent models of stitch recognition

## Overview

insert image

Implementations of recurrent models towards the goal of 

1. Stitch type recognition according to the following defined taxonomy (courtesy of USC neuro-robotic group)

2. Recognition of when a stitch is happening (e.g. "Stitch segmentation")


## Prerequisites

PyTorch (1.3)
ffmpeg 
NVIDIA DALI

Computation using flownet refers to the following github implementation of flownet: 
https://github.com/ClementPinard/FlowNetPytorch

If computing optical flow in a faster way you can use the NVIDIA DALI (https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) framework which uses the turingOF library from nvidia. This is substantially faster inference than flownet but is also coarser in its estimates resulting in slightly lower performance.



## Preprocessing for optical flow

If you have a the mp4 files in a labeled directory with labeleed subdirs

```
-───movie_clips
│   └───00
│       │   fn*.mp4
|       |   ...
│   └───01
│       │   fn*.mp4
|       |   ...
```

Simply run preprocessing/generate_flow.sh to produce versions of the precomputed videos with flow computed either by flownet or DALI.



## Dataset (to be made available soon)

411 labeled video clips of varying length (2-30s) across the 5 gesture types shown above

24 surgical preocedure videos (>1hour) with labeled segments corresponding to times that the needle actually entered tissue

## Results

Models for recognizing and evaluating stitches in srgical videos using deep networks

Currently migrating over from old directory, this will be the newest one as it contains lightning files for expedited training during experiments.
