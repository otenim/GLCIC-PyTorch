import torch.utils.data as data
import os
import torch
import tqdm
import imghdr
import random
from PIL import Image


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir)

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index, color_format='RGB'):
        img = Image.open(self.imgpaths[index])
        img = img.convert(color_format)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
            return True
        else:
            return False

    def __load_imgpaths_from_dir(self, dirpath, walk=False, allowed_formats=None):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, dirs, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if self.__is_imgfile(path) == False:
                    continue
                imgpaths.append(path)
        return imgpaths
