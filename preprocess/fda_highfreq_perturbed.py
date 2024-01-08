#!/usr/bin/env python3

''' Script to precompute features of the high frequency perturbed scenes with ViT, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

import MatterSim

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp


import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from matplotlib import pyplot as plt
import random

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60

# get the low frequency using Gaussian Low-Pass Filter
def gaussian_filter_low_pass(fshift, D):

    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = np.exp(- dis_square / (2 * D ** 2))

    return template * fshift

# get the high frequency using Gaussian High-Pass Filter
def gaussian_filter_high_pass(fshift, D):

    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - np.exp(- dis_square / (2 * D ** 2))

    return template * fshift

# Inverse Fourier transform
def ifft(fshift):

    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifftn(ishift)
    iimg = np.abs(iimg)

    return iimg

def load_viewpoint_ids(connectivity_dir):

    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]

    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))

    return viewpoint_ids

def build_feature_extractor(model_name, checkpoint_file=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=(checkpoint_file is None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device


def build_simulator(connectivity_dir, scan_dir):

    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def get_freq_low_high(img, D):

    f = np.fft.fftn(img)
    fshift = np.fft.fftshift(f)

    high_parts_fshift = gaussian_filter_high_pass(fshift.copy(), D=D)
    low_parts_fshift = gaussian_filter_low_pass(fshift.copy(), D=D)

    return low_parts_fshift, high_parts_fshift

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # random.seed(10)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch ViT model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)

    # 11
    val_unseen_split = ['2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'oLBMNvg9in8', 'QUCTc6BB5sX', 'TbHJrupSAjP',
                       'X7HyMhZNoso', 'pLe4wQe7qrG', 'x8F5xyUWy9e', 'Z6MFQCViBuw', 'zsNo4HB9uLZ']

    # 18
    test_split = ['2t7WUuJeko7', '5ZKStnWn8Zo', 'ARNzJeq3xxb', 'fzynW3qQPVF', 'jtcxE69GiFV', 'pa4otMbVnkk',
                 'q9vSo1VnCiC', 'rqfALeAoiTq', 'UwV83HsGsw3', 'wc2JMjhGNzB', 'WYY7iVyf5p8', 'YFuZgdQ5vWj',
                 'yqstnuAEVhm', 'YVUC4YcDtcY', 'gxdoqLR6rwA', 'gYvKGZ5eRqb', 'RPmz2sHmrrY', 'Vt2qJdWjCF2']

    ###########  sample the interference images and compute their high frequency ############

    random.shuffle(scanvp_list)
    chosen_point_num = 700   # number of interference points (pano-images)
    # freq_high_ = []  # store the low frequency of the interference images
    interference_highfreq = [] # store the low frequency of the interference images
    scans_aug = []

    for scan_id, viewpoint_id in scanvp_list:
        if scan_id in val_unseen_split or scan_id in test_split:
            continue

        if scan_id not in scans_aug:
            scans_aug.append(scan_id)

        # Loop all discretized views from this location
        # [0-11] is looking down, [12-23] is looking at horizon, is [24-35] looking up
        for ix in range(VIEWPOINT_SIZE):

            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])

            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True)  # in BGR channel

            D = 10

            img_1 = image[:, :, 0]
            img_2 = image[:, :, 1]
            img_3 = image[:, :, 2]

            if len(interference_highfreq) < 5 * 3 * chosen_point_num:

                if ix not in [12, 14, 17, 19, 21]:
                    continue

                _, high_parts_fshift_1 = get_freq_low_high(img_1, D)
                _, high_parts_fshift_2 = get_freq_low_high(img_2, D)
                _, high_parts_fshift_3 = get_freq_low_high(img_3, D)

                interference_highfreq.append(high_parts_fshift_1)
                interference_highfreq.append(high_parts_fshift_2)
                interference_highfreq.append(high_parts_fshift_3)

        # 5: select 5 views for each point
        # 3: each view has 3 channels
        # chosen_point_num: number of the chosen point
        if len(interference_highfreq) >= 5 * 3 * chosen_point_num:
            break

    print("number of scans included in the sampled interference images：", len(scans_aug))

    # Mix the frequency and extract the feature
    random.shuffle(scanvp_list)
    for scan_id, viewpoint_id in scanvp_list:

        images_mix_fda = []

        # use the views from the (interference_point_ix)th interference point
        interference_point_ix = random.randint(0, chosen_point_num-1)

        update_signal = random.randint(0, 9)  # whether to update the sampled interference points
        update_ix = random.randint(0, chosen_point_num-1)  # index of the replaced interference points
        while update_ix == interference_point_ix:
            update_ix = random.randint(0, chosen_point_num-1)

        # Loop all discretized views from this location
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            # (480, 640, 3)
            image = np.array(state.rgb, copy=True)  # in BGR channel

            D = 10       # The cutoff frequency

            img_1 = image[:, :, 0]
            img_2 = image[:, :, 1]
            img_3 = image[:, :, 2]

            low_parts_fshift_1, high_parts_fshift_1 = get_freq_low_high(img_1, D)
            low_parts_fshift_2, high_parts_fshift_2 = get_freq_low_high(img_2, D)
            low_parts_fshift_3, high_parts_fshift_3 = get_freq_low_high(img_3, D)

            ###### MIX #######
            cur_low_1 = low_parts_fshift_1
            cur_low_2 = low_parts_fshift_2
            cur_low_3 = low_parts_fshift_3

            cur_high_1 = high_parts_fshift_1
            cur_high_2 = high_parts_fshift_2
            cur_high_3 = high_parts_fshift_3

            # high frequency interference in FDA
            view_in_five = random.randint(0, 4)
            if random.randint(1,3) > 1:
                high_parts_fshift_1 = interference_highfreq[15 * interference_point_ix + view_in_five * 3]

            if random.randint(1, 3) > 1:
                high_parts_fshift_2 = interference_highfreq[15 * interference_point_ix + view_in_five * 3 + 1]

            if random.randint(1, 3) > 1:
                high_parts_fshift_3 = interference_highfreq[15 * interference_point_ix + view_in_five * 3 + 2]

            # update the sampled interference points using current high-frequency
            if scan_id not in val_unseen_split and scan_id not in test_split:
                if update_signal == 0:
                    update_views = [12, 14, 17, 19, 21]

                    for view_id in range(0, 5):
                        if ix == update_views[view_id]:
                            interference_highfreq[update_ix * 15 + 3 * view_id] = cur_high_1
                            interference_highfreq[update_ix * 15 + 3 * view_id + 1] = cur_high_2
                            interference_highfreq[update_ix * 15 + 3 * view_id + 2] = cur_high_3

            # Inversely Fourier Transform in FDA
            i_img_mix1_fda = ifft(cur_low_1 + high_parts_fshift_1)
            i_img_mix1_fda = np.array(i_img_mix1_fda * 1, np.uint8)

            i_img_mix2_fda = ifft(cur_low_2 + high_parts_fshift_2)
            i_img_mix2_fda = np.array(i_img_mix2_fda * 1, np.uint8)

            i_img_mix3_fda = ifft(cur_low_3 + high_parts_fshift_3)
            i_img_mix3_fda = np.array(i_img_mix3_fda * 1, np.uint8)

            i_img_exp_mix1_fda = np.expand_dims(i_img_mix1_fda, axis=2)
            i_img_exp_mix2_fda = np.expand_dims(i_img_mix2_fda, axis=2)
            i_img_exp_mix3_fda = np.expand_dims(i_img_mix3_fda, axis=2)
            i_img_mix_fda = np.append(i_img_exp_mix1_fda, i_img_exp_mix2_fda, axis=2)
            i_img_mix_fda = np.append(i_img_mix_fda, i_img_exp_mix3_fda, axis=2)

            # from BGR to RGB
            i_img_mix_fda = Image.fromarray(i_img_mix_fda[:, :, ::-1])
            images_mix_fda.append(i_img_mix_fda)

        # feature extraction
        images_mix_fda = torch.stack([img_transforms(image).to(device) for image in images_mix_fda], 0)
        fts_fda, logits_fda = [], []
        for k in range(0, len(images_mix_fda), args.batch_size):
            batch_fts = model.forward_features(images_mix_fda[k: k + args.batch_size])
            batch_logits = model.head(batch_fts)
            batch_fts = batch_fts.data.cpu().numpy()
            batch_logits = batch_logits.data.cpu().numpy()
            fts_fda.append(batch_fts)
            logits_fda.append(batch_logits)
        fts_fda = np.concatenate(fts_fda, 0)
        logits_fda = np.concatenate(logits_fda, 0)

        out_queue.put((scan_id, viewpoint_id, fts_fda, logits_fda))

    out_queue.put(None)


def build_feature_file(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)

    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts_fda, logits_fda = res

                key = '%s_%s' % (scan_id, viewpoint_id)

                # True
                if args.out_image_logits:
                    # fts_fda: (36, 768)
                    # logits_fda: (36,1000)
                    data = np.hstack([fts_fda, logits_fda])
                else:
                    data = np.hstack([fts_fda])

                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    build_feature_file(args)

