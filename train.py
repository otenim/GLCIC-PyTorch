from tqdm import tqdm
from models import CompletionNetwork, ContextDiscriminator
from datasets import ImageDataset
from losses import completion_network_loss
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import os
import argparse
import numpy as np
import json


parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('result_dir')
parser.add_argument('--init_model_cn', type=str, default=None)
parser.add_argument('--init_model_cd', type=str, default=None)
parser.add_argument('--steps_1', type=int, default=90000)
parser.add_argument('--steps_2', type=int, default=10000)
parser.add_argument('--steps_3', type=int, default=400000)
parser.add_argument('--snaperiod_1', type=int, default=18000)
parser.add_argument('--snaperiod_2', type=int, default=2000)
parser.add_argument('--snaperiod_3', type=int, default=80000)
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)
parser.add_argument('--cn_input_size', type=int, default=160)
parser.add_argument('--ld_input_size', type=int, default=96)
parser.add_argument('--optimizer', type=str, choices=['adadelta', 'adam'], default='adadelta')
parser.add_argument('--bsize', type=int, default=16)
parser.add_argument('--bdivs', type=int, default=1)
parser.add_argument('--num_gpus', type=int, choices=[1, 2], default=1)
parser.add_argument('--alpha', type=float, default=4e-4)
parser.add_argument('--comp_mpv', default=True)
parser.add_argument('--max_mpv_samples', type=int, default=10000)


def main(args):

    # ================================================
    # Preparation
    # ================================================
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    if args.init_model_cn != None:
        args.init_model_cn = os.path.expanduser(args.init_model_cn)
    if args.init_model_cd != None:
        args.init_model_cd = os.path.expanduser(args.init_model_cd)

    if torch.cuda.is_available() == False:
        raise Exception('At least one gpu must be available.')
    if args.num_gpus == 1:
        # train models in a single gpu
        gpu_cn = torch.device('cuda:0')
        gpu_cd = gpu_cn
    else:
        # train models in different two gpus
        gpu_cn = torch.device('cuda:0')
        gpu_cd = torch.device('cuda:1')

    # create result directory (if necessary)
    if os.path.exists(args.result_dir) == False:
        os.makedirs(args.result_dir)
    for s in ['phase_1', 'phase_2', 'phase_3']:
        if os.path.exists(os.path.join(args.result_dir, s)) == False:
            os.makedirs(os.path.join(args.result_dir, s))

    # dataset
    trnsfm = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset... (it may take a few minutes)')
    train_dset = ImageDataset(os.path.join(args.data_dir, 'train'), trnsfm)
    test_dset = ImageDataset(os.path.join(args.data_dir, 'test'), trnsfm)
    train_loader = DataLoader(train_dset, batch_size=(args.bsize // args.bdivs), shuffle=True)

    # compute the mean pixel value of train dataset
    mean_pv = 0.
    imgpaths = train_dset.imgpaths[:min(args.max_mpv_samples, len(train_dset))]
    if args.comp_mpv:
        pbar = tqdm(total=len(imgpaths), desc='computing the mean pixel value')
        for imgpath in imgpaths:
            img = Image.open(imgpath)
            x = np.array(img, dtype=np.float32) / 255.
            mean_pv += x.mean()
            pbar.update()
        mean_pv /= len(imgpaths)
        pbar.close()
    mpv = torch.tensor(mean_pv).to(gpu_cn)

    # save training config
    args_dict = vars(args)
    args_dict['mean_pv'] = mean_pv
    with open(os.path.join(args.result_dir, 'config.json'), mode='w') as f:
        json.dump(args_dict, f)


    # ================================================
    # Training Phase 1
    # ================================================
    # model & optimizer
    model_cn = CompletionNetwork()
    if args.init_model_cn != None:
        model_cn.load_state_dict(torch.load(args.init_model_cn, map_location='cpu'))
    if args.optimizer == 'adadelta':
        opt_cn = Adadelta(model_cn.parameters())
    else:
        opt_cn = Adam(model_cn.parameters())
    model_cn = model_cn.to(gpu_cn)

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1:
        for x in train_loader:

            # generate hole area
            hole_area = gen_hole_area(
                size=(args.ld_input_size, args.ld_input_size),
                mask_size=(x.shape[3], x.shape[2]),
            )

            # create mask
            msk = gen_input_mask(
                shape=x.shape,
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h),
                ),
                hole_area=hole_area,
                max_holes=args.max_holes,
            )

            # merge x, mask, and mpv
            msg = 'phase 1 |'
            x = x.to(gpu_cn)
            msk = msk.to(gpu_cn)
            input = x - x * msk + mpv * msk
            output = model_cn(input)

            # backward
            loss = completion_network_loss(x, output, msk)
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs > args.bdivs:
                
                # optimize
                opt_cn.step()
                cnt_bdivs = 0

                # clear grads
                opt_cn.zero_grad()

                # update progbar
                pbar.set_description(' train loss: %.5f' % loss.cpu())
                pbar.update()

                # test
                if pbar.n % args.snaperiod_1 == 0:
                with torch.no_grad():
                    x = sample_random_batch(test_dset, batch_size=args.bsize)
                    x = x.to(gpu_cn)
                    input = x - x * msk + mpv * msk
                    output = model_cn(input)
                    completed = poisson_blend(input, output, msk)
                    imgs = torch.cat((input.cpu(), completed.cpu()), dim=0)
                    save_image(imgs, os.path.join(args.result_dir, 'phase_1', 'step%d.png' % pbar.n), nrow=len(x))
                    torch.save(model_cn.state_dict(), os.path.join(args.result_dir, 'phase_1', 'model_cn_step%d' % pbar.n))

                if pbar.n >= args.steps_1:
                    break
    pbar.close()


    # ================================================
    # Training Phase 2
    # ================================================
    # model, optimizer & criterion
    model_cd = ContextDiscriminator(
        local_input_shape=(3, args.ld_input_size, args.ld_input_size),
        global_input_shape=(3, args.cn_input_size, args.cn_input_size),
    )
    if args.init_model_cd != None:
        model_cd.load_state_dict(torch.load(args.init_model_cd, map_location='cpu'))
    if args.optimizer == 'adadelta':
        opt_cd = Adadelta(model_cd.parameters())
    else:
        opt_cd = Adam(model_cd.parameters())
    model_cd = model_cd.to(gpu_cd)
    bceloss = BCELoss()

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_2)
    while pbar.n < args.steps_2:
        for x in train_loader:

            x = x.to(gpu_cn)

            # ================================================
            # fake
            # ================================================
            hole_area = gen_hole_area(
                size=(args.ld_input_size, args.ld_input_size),
                mask_size=(x.shape[3], x.shape[2]),
            )

            # create mask
            msk = gen_input_mask(
                shape=x.shape,
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h),
                ),
                hole_area=hole_area,
                max_holes=args.max_holes,
            )

            fake = torch.zeros((len(x), 1)).to(gpu_cd)
            msk = msk.to(gpu_cn)
            input_cn = x - x * msk + mpv * msk
            output_cn = model_cn(input_cn)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, hole_area)
            input_fake = (input_ld_fake.to(gpu_cd), input_gd_fake.to(gpu_cd))
            output_fake = model_cd(input_fake)
            loss_fake = bceloss(output_fake, fake)

            # ================================================
            # real
            # ================================================
            hole_area = gen_hole_area(
                size=(args.ld_input_size, args.ld_input_size),
                mask_size=(x.shape[3], x.shape[2]),
            )

            real = torch.ones((len(x), 1)).to(gpu_cd)
            input_gd_real = x
            input_ld_real = crop(input_gd_real, hole_area)
            input_real = (input_ld_real.to(gpu_cd), input_gd_real.to(gpu_cd))
            output_real = model_cd(input_real)
            loss_real = bceloss(output_real, real)

            # ================================================
            # optimize
            # ================================================
            loss = (loss_fake + loss_real) / 2.
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs < args.bdivs:
                continue
            cnt_bdivs = 0
            opt_cd.step()
            opt_cd.zero_grad()

            msg = 'phase 2 |'
            msg += ' train loss: %.5f' % loss.cpu()
            pbar.set_description(msg)
            pbar.update()

            # test
            if pbar.n % args.snaperiod_2 == 0:
                with torch.no_grad():

                    x = sample_random_batch(test_dset, batch_size=args.bsize)
                    x = x.to(gpu_cn)
                    input = x - x * msk + mpv * msk
                    output = model_cn(input)
                    completed = poisson_blend(input, output, msk)
                    imgs = torch.cat((input.cpu(), completed.cpu()), dim=0)
                    save_image(imgs, os.path.join(args.result_dir, 'phase_2', 'step%d.png' % pbar.n), nrow=len(x))
                    torch.save(model_cd.state_dict(), os.path.join(args.result_dir, 'phase_2', 'model_cd_step%d' % pbar.n))

            if pbar.n >= args.steps_2:
                break
    pbar.close()


    # ================================================
    # Training Phase 3
    # ================================================
    # training
    cnt_bdivs = 0
    alpha = torch.tensor(args.alpha).to(gpu_cd)
    pbar = tqdm(total=args.steps_3)
    while pbar.n < args.steps_3:
        for x in train_loader:

            x = x.to(gpu_cn)

            # ================================================
            # train model_cd
            # ================================================
            # fake
            hole_area = gen_hole_area(
                size=(args.ld_input_size, args.ld_input_size),
                mask_size=(x.shape[3], x.shape[2]),
            )

            # create mask
            msk = gen_input_mask(
                shape=x.shape,
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h),
                ),
                hole_area=hole_area,
                max_holes=args.max_holes,
            )

            fake = torch.zeros((len(x), 1)).to(gpu_cd)
            msk = msk.to(gpu_cn)
            input_cn = x - x * msk + mpv * msk
            output_cn = model_cn(input_cn)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, hole_area)
            input_fake = (input_ld_fake.to(gpu_cd), input_gd_fake.to(gpu_cd))
            output_fake = model_cd(input_fake)
            loss_cd_1 = bceloss(output_fake, fake)

            # real
            hole_area = gen_hole_area(
                size=(args.ld_input_size, args.ld_input_size),
                mask_size=(x.shape[3], x.shape[2]),
            )

            real = torch.ones((len(x), 1)).to(gpu_cd)
            input_gd_real = x
            input_ld_real = crop(input_gd_real, hole_area)
            input_real = (input_ld_real.to(gpu_cd), input_gd_real.to(gpu_cd))
            output_real = model_cd(input_real)
            loss_cd_2 = bceloss(output_real, real)

            # optimize
            loss_cd = (loss_cd_1 + loss_cd_2) * alpha / 2.
            loss_cd.backward()
            cnt_bdivs += 1
            if cnt_bdivs < args.bdivs:
                pass
            else:
                opt_cd.step()
                opt_cd.zero_grad()

            # ================================================
            # train model_cn
            # ================================================
            loss_cn_1 = completion_network_loss(x, output_cn, msk).to(gpu_cd)
            input_gd_fake = output_cn
            input_ld_fake = crop(input_gd_fake, hole_area)
            input_fake = (input_ld_fake.to(gpu_cd), input_gd_fake.to(gpu_cd))
            output_fake = model_cd(input_fake)
            loss_cn_2 = bceloss(output_fake, real)

            # optimize
            loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.
            loss_cn.backward()
            if cnt_bdivs < args.bdivs:
                continue
            cnt_bdivs = 0
            opt_cn.step()
            opt_cn.zero_grad()
            msg = 'phase 3 |'
            msg += ' train loss (cd): %.5f' % loss_cd.cpu()
            msg += ' train loss (cn): %.5f' % loss_cn.cpu()
            pbar.set_description(msg)
            pbar.update()

            # test
            if pbar.n % args.snaperiod_3 == 0:
                with torch.no_grad():

                    x = sample_random_batch(test_dset, batch_size=args.bsize)
                    x = x.to(gpu_cn)
                    input = x - x * msk + mpv * msk
                    output = model_cn(input)
                    completed = poisson_blend(input, output, msk)
                    imgs = torch.cat((input.cpu(), completed.cpu()), dim=0)
                    save_image(imgs, os.path.join(args.result_dir, 'phase_3', 'step%d.png' % pbar.n), nrow=len(x))
                    torch.save(model_cn.state_dict(), os.path.join(args.result_dir, 'phase_3', 'model_cn_step%d' % pbar.n))
                    torch.save(model_cd.state_dict(), os.path.join(args.result_dir, 'phase_3', 'model_cd_step%d' % pbar.n))

            if pbar.n >= args.steps_3:
                break
    pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
