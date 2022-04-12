#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:26:36 2021

@author: xingw
"""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='RSTCANet_B', help='RSTCANet_B or RSTCANet_S or RSTCANet_L')
    parser.add_argument('--testset_name', type=str, default='McM', help='McM or kodak or CBSD68 or urban100')
    parser.add_argument('--need_degradation', type=bool, default=True, help='Data preprocessing')
    parser.add_argument('--show_img', type=bool, default=False, help='Show the mosaiced, demosaiced and GT images')
    parser.add_argument('--task_current', type=str, default='dm_model_zoo', help='dm_model_zoo')
    parser.add_argument('--pattern', type=int, default=1, help='1 (RGGB) or 2 (GRBG) or 3 (GBRG) or 4 (BGGR)')
    parser.add_argument('--n_channels', type=int, default=3, help='setting for color image')
    parser.add_argument('--nc', type=int, default=72, help='channel number')
    parser.add_argument('--window_size', type=int, default=8, help='window size of Swin Transformer')
    parser.add_argument('--num_heads', type=int, default=6, help='window size of Swin Transformer')
    parser.add_argument('--K', type=int, default=6, help='number of STLs in one RSTCAB')
    parser.add_argument('--N', type=int, default=2, help='number of RSTCAB')
    parser.add_argument('--patch_size', type=int, default=2, help='patch size for patch partition')
    parser.add_argument('--sf', type=int, default=0, help='scale factor')
    
    args = parser.parse_args()

    num_heads=[args.num_heads for i in range(args.N)]
    depths = [args.K for i in range(args.N)]
    
    sf = args.sf
    
    if 'clip' in args.model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False
    testsets = 'testset'     # fixed
    results = 'results'       # fixed
    result_name = args.testset_name + '_' + args.model_name
    border = sf if args.task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(args.task_current, args.model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, args.testset_name) # L_path, for Low-quality images
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
    model = net(in_nc=1, out_nc=args.n_channels, patch_size=args.patch_size, nc=args.nc, 
                window_size=args.window_size, num_heads=num_heads, depths=depths)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}'.format(args.model_name))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=args.n_channels)

        if need_degradation:  # degradation process
#            img_L, h_border, w_border = util.image_padding(img_L, 16*4)
            img_L = util.modcrop(img_L, 16)
            img_L = util.mosaic_CFA_Bayer(img_L, args.pattern)
            # # demosaicing
            # img_L = util.demosaic(img_L, pattern)
            if use_clip:
                img_L = util.uint2single(img_L)

        util.imshow(img_L, title='Mosaic image') if args.show_img else None

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
            img_H = util.imread_uint(H_paths[idx], n_channels=args.n_channels)
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
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if args.show_img else None

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
