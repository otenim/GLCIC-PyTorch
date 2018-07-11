from utils import add_random_patches, gen_random_patch_region
import torch

x = torch.zeros(1, 1, 10, 10)
preg = gen_random_patch_region((10, 10), region_size=(5,5))
x[0, 0, preg[0][1] : preg[0][1] + preg[1][1], preg[0][0] : preg[0][0] + preg[1][0]] = 2.0
msk = add_random_patches(x,
    patch_size=((2,4), (2,4)),
    patch_region=preg,
)

print(msk)
print(preg)
