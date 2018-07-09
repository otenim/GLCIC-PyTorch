import torch
import random

def generate_random_mask(shape, patch_size=((16, 48), (16, 48)), max_patches=3):
    bsize = shape[0]
    msks = []
    for i in range(bsize):
        msk = torch.zeros(shape[1:])
        _, msk_h, msk_w = msk.shape
        n_patches = random.choice(list(range(1, max_patches+1)))
        for j in range(n_patches):
            # choose patch height
            if isinstance(patch_size[0], tuple) and len(patch_size[0]) == 2:
                min_patch_h, max_patch_h = patch_size[0]
                patch_h = random.randint(min_patch_h, max_patch_h)
            else:
                patch_h = patch_size[0]

            # choose patch width
            if isinstance(patch_size[1], tuple) and len(patch_size[1]) == 2:
                min_patch_w, max_patch_w = patch_size[1]
                patch_w = random.randint(min_patch_w, max_patch_w)
            else:
                patch_w = patch_size[1]

            # choose offset upper-left coordinat
            offset_x = random.randint(0, msk_w - patch_w)
            offset_y = random.randint(0, msk_h - patch_h)
            msk[:, offset_y:offset_y + patch_h, offset_x:offset_x + patch_w] = 1.0
        msks.append(msk.unsqueeze(dim=0))
    return torch.cat(msks, dim=0)
