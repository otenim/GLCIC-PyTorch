import torch
import random


def add_random_patches(
    mask, patch_size,
    patch_region=None, max_patches=1):
    """
    * inputs:
        - mask (pytorch tensor, required):
                A (samples, c, h, w) format pytorch tensor.
                its values are filled with 0.
        - patch_size (sequence or int, required):
                Desired size of pathces.
                If a sequence of length 4 provided,
                patches of size (w, h) = (
                    patch_size[0][0] <= patch_size[0][1],
                    patch_size[1][0] <= patch_size[1][1],
                ) are generated.
                All the patched pixel values are filled with 1.
        - patch_region (sequence, optional):
                A region where pathces are randomly generated.
                patch_region[0] means the left corner (x, y) of the region,
                and patch_region[1] mean the width and height (w, h) of it.
                This is used as an input region of Local discriminator.
                The default value is None.
        - max_patches (int, optional):
                It specifies how many patches are generated.
                The default value is 1.
    * returns:
            Input mask tensor with generated pathces
            where all the pixel values are filled with 1.
    """

    mask = mask.clone()
    bsize, _, mask_h, mask_w = mask.shape
    masks = []
    for i in range(bsize):
        n_patches = random.choice(list(range(1, max_patches+1)))
        for j in range(n_patches):
            # choose patch width
            if isinstance(patch_size[0], tuple) and len(patch_size[0]) == 2:
                patch_w = random.randint(patch_size[0][0], patch_size[0][1])
            else:
                patch_w = patch_size[0]

            # choose patch height
            if isinstance(patch_size[1], tuple) and len(patch_size[1]) == 2:
                patch_h = random.randint(patch_size[1][0], patch_size[1][1])
            else:
                patch_h = patch_size[1]

            # choose offset upper-left coordinate
            if patch_region:
                preg_xmin, preg_ymin = patch_region[0]
                preg_w, preg_h = patch_region[1]
                offset_x = random.randint(preg_xmin, preg_xmin + preg_w - patch_w)
                offset_y = random.randint(preg_ymin, preg_ymin + preg_h - patch_h)
            else:
                offset_x = random.randint(0, mask_w - patch_w)
                offset_y = random.randint(0, mask_h - patch_h)
            mask[i, :, offset_y : offset_y + patch_h, offset_x : offset_x + patch_w] = 1.0
    return mask


def gen_random_patch_region(mask_size, region_size):
    """
    * inputs:
        - mask_size (sequence, required)
                The size of an inputs mask tensor.
        - region_size (sequence, required)
                The size of a region where patches are generated.
    * returns:
            A random region of size (w, h) = (region_size[0], region_size[1])
            where patches are generated.
            returns[0] means the left corner (x, y) of the region,
            and returns[1] mean the size (w, h) of the region.
            This sequence is used as an input argument of add_random_patches function.
            The region is randomly generated within the input mask.
    """
    mask_w, mask_h = mask_size
    region_w, region_h = region_size
    offset_x = random.randint(0, mask_w - region_w)
    offset_y = random.randint(0, mask_h - region_h)
    return ((offset_x, offset_y), (region_w, region_h))


def crop_patch_region(x, patch_region):
    """
    * inputs:
        - x (Tensor, required)
                A pytorch 4D tensor (samples, c, h, w).
        - patch_region (sequence, required)
                A patch region ((x_min, y_min), (w, h)).
    * returns:
            A pytorch tensor cropped in the input patch region.
    """
    xmin, ymin = patch_region[0]
    w, h = patch_region[1]
    return x[:, :, ymin : ymin + h, xmin : xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for i in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def save_args(filepath, args):
    """
    * inputs:
        - filepath (path, required)
                A path to the file where 'args' is to be dumped.
        - args (argparse.Namespace, required)
                An output object of argparse.ArgumentParser().parse_args()
    * returns:
            A text file where the output object of argparse.ArgumentParser()
            is dumpesd is created in the place specified with 'filepath'.
    """
    args_dict = vars(args)
    with open(filepath, mode='w') as f:
        for key in args_dict.keys():
            f.write('%s: %s\n' % (key, args_dict[key]))
