import torch
import random


def add_random_patches(
    mask, patch_size,
    patch_region=None, max_patches=1):

    bsize = mask.shape[0]
    masks = []
    for i in range(bsize):
        _, mask_h, mask_w = mask.shape
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
                pregion_xmin, pregion_ymin = patch_region[0]
                pregion_w, pregion_h = patch_region[1]
                offset_x = random.randint(pregion_xmin, pregion_w - patch_w)
                offset_y = random.randint(pregion_ymin, pregion_h - patch_h)
            else:
                offset_x = random.randint(0, mask_w - patch_w)
                offset_y = random.randint(0, mask_h - patch_h)
            mask[:, offset_y:offset_y + patch_h, offset_x:offset_x + patch_w] = 1.0
        masks.append(mask.unsqueeze(dim=0))
    return torch.cat(masks, dim=0)


def gen_random_patch_region(mask_size, region_size):
    mask_w, mask_h = mask_size
    region_w, region_h = region_size
    offset_x = random.randint(0, mask_w - region_w)
    offset_y = random.randint(0, mask_h - region_h)
    return ((offset_x, offset_y), (region_w, region_h))
