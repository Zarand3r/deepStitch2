import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from gluoncv.torch.utils.utils import build_log_dir
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model
from dataloader import VideoClsDataset


def main(cfg, save_path):

    # get model
    print('Building model for feature extraction')
    model = get_model(cfg)
    model.cuda()
    model.eval()
    print('Pre-trained model is successfully loaded from the model zoo.')

    # get data
    val_dataset = VideoClsDataset(anno_path=cfg.CONFIG.DATA.VAL_ANNO_PATH,
                                  data_path=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                  mode='features',
                                  use_multigrid=cfg.CONFIG.DATA.MULTIGRID,
                                  clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                  frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                  num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                                  num_crop=cfg.CONFIG.DATA.NUM_CROP,
                                  keep_aspect_ratio=cfg.CONFIG.DATA.KEEP_ASPECT_RATIO,
                                  crop_size=cfg.CONFIG.DATA.CROP_SIZE,
                                  short_side_size=cfg.CONFIG.DATA.SHORT_SIDE_SIZE,
                                  new_height=cfg.CONFIG.DATA.NEW_HEIGHT,
                                  new_width=cfg.CONFIG.DATA.NEW_WIDTH)

    samples = val_dataset.dataset_samples
    print('Extracting features from %d videos.' % len(val_dataset))
    start_time = time.time()
    for vid, vtuple in enumerate(val_dataset):
        video_clip, video_label, video_name = vtuple
        # the line below was a bootstrap necessary before the bug was fixed. Can comment it out now
        video_name = os.path.basename(samples[vid]).split(".")[0]
        video_features = []
        video_clip = torch.unsqueeze(video_clip, dim=0).cuda()
        print(video_name)
        print(video_clip.shape)

        for i in range(video_clip.shape[2]-5):
            video_frame = video_clip[:, :, i:i+5, :, :]
            with torch.no_grad():
                feat = model(video_frame).cpu().numpy()
                video_features.append(feat)
        video_features = np.squeeze(video_features)

        feat_file = '%s.npy' % (video_name)
        np.save(os.path.join(save_path, feat_file), video_features)

        if vid > 0 and vid % 10 == 0:
            print('%04d/%04d is done' % (vid, len(val_dataset)))

    end_time = time.time()
    print('Total feature extraction time is %4.2f minutes' % ((end_time - start_time) / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features.')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.CONFIG.LOG.EXP_NAME = ""
    _ = build_log_dir(cfg)
    save_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.SAVE_DIR)
    print('Saving extracted features to %s' % save_path)
    cfg.freeze()

    main(cfg, save_path)
