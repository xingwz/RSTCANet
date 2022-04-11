#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:26:36 2021

@author: xingw
"""

import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Wenzhu Xing Modify from
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/FFDNet

@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'RSTCANet_L'           # 'RSTCANet_B' | 'RSTCANet_S' | 'RSTCANet_L'
    testset_name = 'McM'               # test set,  'McM' | 'kodak' | 'CBSD68' | 'urban100'
    need_degradation = True              # default: True
    show_img = False                     # default: False




    task_current = 'dm_model_zoo'
    pattern = 1                    # unused for Bayer pattern
    n_channels = 3        # setting for color image
    nc = 128               # setting for color image
    window_size=8
    num_heads=[8, 8, 8, 8]
    depths = [8, 8, 8, 8]
    patch_size = 2               # setting for color image
    
    sf = 1
    
    if 'clip' in model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False
    model_pool = 'models'  # fixed
    testsets = 'testset'     # fixed
    results = 'results'       # fixed
    result_name = testset_name + '_' + model_name
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(task_current, model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_dm import RSTCANet as net
    model = net(in_nc=1, out_nc=n_channels, patch_size=patch_size, nc=nc, 
                window_size=window_size, num_heads=num_heads, depths=depths)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}'.format(model_name))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)

        if need_degradation:  # degradation process
#            img_L, h_border, w_border = util.image_padding(img_L, 16*4)
            img_L = util.modcrop(img_L, 16)
            img_L = util.mosaic_CFA_Bayer(img_L, pattern)
            # # demosaicing
            # img_L = util.demosaic(img_L, pattern)
            if use_clip:
                img_L = util.uint2single(img_L)

        util.imshow(img_L, title='Mosaic image') if show_img else None

        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        img_E = model(img_L)
        img_E = util.tensor2uint(img_E)
#        img_E = util.shave_two(img_E, h_border, w_border)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = util.modcrop(img_H, 16)
            img_H = img_H.squeeze()

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
