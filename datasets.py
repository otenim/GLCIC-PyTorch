import torch.utils.data as data
import os
import torch
import tqdm
from PIL import Image
from datasetutils.image import load_imgpaths_from_dir

class ImageDataset(data.Dataset):
    def __init__(self, root, transform=None):
        super(ImageDataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.imgpaths = load_imgpaths_from_dir(self.root)

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index):
        img = Image.open(self.imgpaths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img
