import os
import argparse
import imghdr
import random
import shutil
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--split', type=float, default=0.8)


def main(args):
    args.data_dir = os.path.expanduser(args.data_dir)

    print('loading dataset...')
    src_paths = []
    for file in os.listdir(args.data_dir):
        path = os.path.join(args.data_dir, file)
        if imghdr.what(path) == None:
            continue
        src_paths.append(path)
    random.shuffle(src_paths)

    # separate the paths
    border = int(args.split * len(src_paths))
    train_paths = src_paths[:border]
    test_paths = src_paths[border:]
    print('train images: %d images.' % len(train_paths))
    print('test images: %d images.' % len(test_paths))

    # create dst directories
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    if os.path.exists(train_dir) == False:
        os.makedirs(train_dir)
    if os.path.exists(test_dir) == False:
        os.makedirs(test_dir)

    # move the image files
    pbar = tqdm.tqdm(total=len(src_paths))
    for dset_paths, dset_dir in zip([train_paths, test_paths], [train_dir, test_dir]):
        for src_path in dset_paths:
            dst_path = os.path.join(dset_dir, os.path.basename(src_path))
            shutil.move(src_path, dst_path)
            pbar.update()
    pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
