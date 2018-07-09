from tqdm import tqdm
from models import CompletionNetwork, ContextDiscriminator
from datasets import ImageDataset
from losses import completion_network_loss
from torch.utils.data import DataLoader
from torch.optim import Adadelta
import torchvision.transforms as transforms
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--Tc', type=int, default=9000)
parser.add_argument('--Td', type=int, default=1000)
parser.add_argument('--Ttrain', type=int, default=50000)
parser.add_argument('--model_c_img_h', type=int, default=218)
parser.add_argument('--model_c_img_w', type=int, default=178)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--shuffle', default=True)

def main(args):

    args.data_dir = os.path.expanduser(args.data_dir)

    # ================================================
    # Training Phase 1
    # ================================================
    trnsfm_1 = transforms.Compose([
        transforms.Resize((args.model_c_img_h, args.model_c_img_w)),
        transforms.ToTensor(),
    ])

    train_dset_1 = ImageDataset(os.path.join(args.data_dir, 'train'), trnsfm_1)
    valid_dset_1 = ImageDataset(os.path.join(args.data_dir, 'valid'), trnsfm_1)
    train_loader_1 = DataLoader(train_dset_1, batch_size=args.batch_size, shuffle=args.shuffle)
    valid_loader_1 = DataLoader(valid_dset_1, batch_size=args.batch_size, shuffle=args.shuffle)

    model_c = CompletionNetwork()
    opt_c = Adadelta(model_c.parameters())

    n = 0
    pbar = tqdm(total=args.Tc, desc='training phase 1')
    while n < args.Tc:
        for x in train_loader_1:

            opt_c.zero_grad()
            msk = generate_completion_mask()
            y = model_c(x)
            loss = completion_network_loss(x, y, msk)
            loss.backward()
            opt_c.step()

            n += 1
            pbar.update()
            if n >= args.Tc:
                break
    # ================================================
    # Training Phase 2
    # ================================================

    # ================================================
    # Training Phase 3
    # ================================================


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
