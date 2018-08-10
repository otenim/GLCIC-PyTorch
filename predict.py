import os
import argparse
import torch
import json
import numpy
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import CompletionNetwork
from PIL import Image
from utils import poisson_blend, gen_input_mask


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_img')
parser.add_argument('output_img')
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)


def main(args):

    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_img = os.path.expanduser(args.input_img)
    args.output_img = os.path.expanduser(args.output_img)


    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = config['mean_pv']
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))


    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    img = Image.open(args.input_img)
    img = transforms.Resize(args.img_size)(img)
    img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    # create mask
    msk = gen_input_mask(
        shape=x.shape,
        hole_size=(
            (args.hole_min_w, args.hole_max_w),
            (args.hole_min_h, args.hole_max_h),
        ),
        max_holes=args.max_holes,
    )

    # inpaint
    with torch.no_grad():
        input = x - x * msk + mpv * msk
        output = model(input)
        inpainted = poisson_blend(input, output, msk)
        imgs = torch.cat((x, input, inpainted), dim=-1)
        imgs = save_image(imgs, args.output_img, nrow=3)
    print('output img was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
