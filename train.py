from models import CompletionNetwork, ContextDiscriminator
from datasets import ImageDataset
from losses import completion_network_loss
from utils import add_random_patches, gen_random_patch_region
from torch.utils.data import DataLoader
from torch.optim import Adadelta
import torch
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--Tc', type=int, default=9000)
parser.add_argument('--Td', type=int, default=1000)
parser.add_argument('--Ttrain', type=int, default=50000)
parser.add_argument('--max_patches', type=int, default=1)
parser.add_argument('--ptch_reg_w', type=int, default=96)
parser.add_argument('--ptch_reg_h', type=int, default=96)
parser.add_argument('--ptch_min_w', type=int, default=24)
parser.add_argument('--ptch_max_w', type=int, default=72)
parser.add_argument('--ptch_min_h', type=int, default=24)
parser.add_argument('--ptch_max_h', type=int, default=72)
parser.add_argument('--cn_input_size', type=int, default=160)
parser.add_argument('--bsize', type=int, default=16)
parser.add_argument('--shuffle', default=True)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--rho', type=float, default=0.9)
parser.add_argument('--wd', type=float, default=0.0)


def main(args):

    args.data_dir = os.path.expanduser(args.data_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    # ================================================
    # Training Phase 1
    # ================================================
    trnsfm_1 = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])

    # dataset
    train_dset_1 = ImageDataset(os.path.join(args.data_dir, 'train'), trnsfm_1)
    valid_dset_1 = ImageDataset(os.path.join(args.data_dir, 'valid'), trnsfm_1)
    train_loader_1 = DataLoader(train_dset_1, batch_size=args.bsize, shuffle=args.shuffle, **kwargs)
    valid_loader_1 = DataLoader(valid_dset_1, batch_size=args.bsize, shuffle=args.shuffle, **kwargs)

    # compute the mean pixe; value of datasets
    imgpaths = train_dset_1.imgpaths + valid_dset_1.imgpaths
    pbar = tqdm(total=len(imgpaths), desc='computing the mean pixel value of datasets')
    mpv = 0.
    for imgpath in imgpaths:
        img = Image.open(imgpath)
        x = np.array(img, dtype=np.float32)
        mpv += x.mean()
        pbar.update()
    pbar.close()
    mpv /= len(imgpaths)
    mpv /= 255. # normalize

    # model & optimizer
    model_cn = CompletionNetwork().to(device)
    opt_cn = Adadelta(model_cn.parameters(), lr=args.lr, rho=args.rho, weight_decay=args.wd)

    # training
    pbar = tqdm(total=args.Tc, desc='phase 1')
    while pbar.n < args.Tc:
        for x in train_loader_1:

            x.to(device)
            opt_cn.zero_grad()

            # generate patch region
            ptch_reg = gen_random_patch_region(
                mask_size=(x.shape[3], x.shape[2]),
                region_size=(args.ptch_reg_w, args.ptch_reg_h),
            )

            # create mask
            msk = add_random_patches(
                torch.zeros_like(x),
                patch_size=(
                    (args.ptch_min_w, args.ptch_max_w),
                    (args.ptch_min_h, args.ptch_max_h)),
                patch_region=ptch_reg,
                max_patches=args.max_patches,
            )

            # merge x, mask, and mpv
            msg = 'phase 1|'
            input = x - x * msk + mpv * msk
            output = model_cn(input)
            loss = completion_network_loss(x, output, msk)
            loss.backward()
            opt_cn.step()

            msg += ' train loss: %.5f' % loss
            pbar.set_description(msg)
            pbar.update()
            if pbar.n >= args.Tc:
                break
    pbar.close()

    # ================================================
    # Training Phase 2
    # ================================================

    # ================================================
    # Training Phase 3
    # ================================================


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
