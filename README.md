# GLCIC-pytorch


## About this repository

Here, we provide a high-quality pytorch implementation of [GLCIC](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) introduced by Iizuka et. al.

![glcic](https://i.imgur.com/KY26J85.png)  
![result_1](https://i.imgur.com/SYkn6Uo.png)  
![result_2](https://i.imgur.com/T8GGx1g.jpg)  

## Dependencies

We tested our scripts in the following environment.

* Python: 3.5 or 3.6
* torch: 0.4.1 or 0.4.0
* torchvision: 0.2.1
* tqdm: 4.24.0
* Pillow: 5.2.0
* numpy: 1.15.0
* pyamg: **3.3.2 (important)**
* scipy: 1.1.0
* GPU: Geforce GTX 1080Ti (12GB RAM) X 1

If you would like to run our training script, we recommend you to
use a more than middle-range GPU such as GTX 1070 or GTX 1080(Ti).

## DEMO (Inference)

### 1. Download our pretrained model and the training config file.

**For Python 3.5 Users**  
Pretrained model: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/Eaosyb919AJPjYau4ALWmKUB2i0L1lVh0dqVxhB2aHwBhg?e=GQBkP4)  
Training config file: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/Ebu3pP2wG2FKt1rRZzF_yEkBsVgxkBdJ28poeDfmGTz3aA?e=gpoAGc)

**For Python 3.6 Users**  
Pretrained model: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EXPGbI_yvFNIhXNI7WgtgMkBdMbxJAdWJWbI5hNBJtHWUg?e=OYWDlH)  
Training config file: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/Ed0myTTrjN9FiX8sYwr4dsYBOEyj3pH_EQbu31HadjUvlw?e=DwAwaO)

The pretrained model was trained with [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset,  
and the training config file saves training settings in json format.


### 2. Inference

Left: input image  
Center: input image of GLCIC  
Right: output image of GLCIC  

![predict_1](https://i.imgur.com/U4VAeFc.jpg)  
![predict_2](https://i.imgur.com/B4T8Z3Y.jpg)  
![predict_3](https://i.imgur.com/1wRQf5m.jpg)  

```bash
# in ***/GLCIC-pytorch/,
$ python predict.py model_cn_step400000 config.json images/test_1.jpg out.jpg
```

## DEMO (Training)

Here, we introduce how to train a model using CelebA dataset.

### 1. Download the dataset

First, download the dataset (i.e., img_align_celeba.zip) from [this official link](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM).

Second, run the following commands.

```bash
$ unzip img_align_celeba.zip
$ mv img_align_celeba/ ***/GLCIC-pytorch/datasets/
$ cd ***/GLCIC-pytorch/datasets/
$ python make_dataset.py img_align_celeba/
```

Originally, all the images are stored in `img_align_celeba/`,
and the last command splits the dataset into two subsets; training dataset (80%) and test dataset (20%). All the training images are stored in `img_align_celeba/train/`, while
the other images are in `img_align_celeba/test/`.


### 2. Training

Just run the following command.

```
# in ***/GLCIC-pytorch/
$ python train.py datasets/img_align_celeba results/demo/
```

Training results (trained models and inference results at each snapshot period) are to be saved in `results/demo/`.

The training procedure consists the following three phases.  
1. In phase 1, only Completion Network (i.e., generator) is trained.
2. In phase 2, only Contect Discriminator (i.e., discriminator) is trained, while Completion Network is frozen.
3. In phase 3, Both of the Completion Network and the Context Discriminator are jointly trained.

Under default settings, the numbers of training steps during phase 1, phase 2, and phase 3 are 90,000, 10,000, and 400,000. Each snapshot period is set to 18,000, 2,000, and 80,000, respectively. Bach size is set to 16, while input image size is 160 x 160 (all the input images are rescalled so that the minumum side is 160, then randomly cropped to 160 x 160 images).

Basically, hyper parameters and the model architecture is exactly the same as described in the paper, but we changed the batch size from 96 to 16 due to lack of GPU memory.

## How to train with your own dataset ?

### 1. Prepare dataset

To train a model with your own dataset, first you have to make a dataset
directory in the following format.

```
dataset/ # the directory name can be anything.
    |____train/
    |       |____XXXX.jpg # png images are also OK.
    |       |____OOOO.jpg
    |____test/
            |____oooo.jpg
            |____xxxx.jpg  
```

Images in `dataset/train/` are used for training models, while
images in `dataset/test/` are used to perform test inpainting at each
snapshot period.

### 2. Training

```bash
# in ***/GLCIC-pytorch/
$ mv dataset/ datasets/
$ python train.py datasets/dataset/ results/result/ [--cn_input_size] [--ld_input_size] [--steps_1] [--steps_2] [--steps_3] [--snaperiod_1] [--snaperiod_2] [--snaperiod_3] [--bsize]
```

**Arguments**  
* `--cn_input_size`: Input size of Completion Network (default: 160). All the input images are rescalled so that the length of the minimum side = cn_input_size,
then cropped to cn_input_size x cn_input_size images.
* `--ld_input_size`: Input size of Local Discriminator (default: 96).
* `--steps_1`: Training iterations in phase 1 (default: 90,000).
* `--steps_2`: Training iterations in phase 2 (default: 10,000).
* `--steps_3`: Training iterations in the last phase (default: 400,000).
* `--snaperiod_1`: Snapshot period in phase 1 (default: 18,000).
* `--snaperiod_2`: Snapshot period in phase 2 (default: 2,000).
* `--snaperiod_3`: Snapshot period in the last phase (default: 80,000).
* `--bsize`: Batch size (default: 16).

**Example**: If you'd like to train a model with batch size 24, and the other parameters are default values, run the following command.

```bash
# in ***/GLCIC-pytorch/
$ python train.py datasets/dataset results/result --bsize 24
```

## How to infer with your own dataset ?

Suppose you've finished train a model and the result directory is set to `***/GLCIC-pytorch/results/result`, run the following command.

```bash
# in ***/GLCIC-pytorch/
$ python predict.py results/result/phase_*/model_cn_step* results/result/config.json <input_img> <output_img> [--max_holes] [--img_size] [--hole_min_w] [--hole_max_w] [--hole_min_h] [--hole_max_h]
```

**Arguments**  
* `<input_img>` (required): Path to an input image.
* `<output_img>` (required): Path to an output image.
* `[--max_holes]`: The max number of holes (default: 1).
* `[--img_size]`: Input size of Completion Network (default: 160). The input image are rescalled so that the length of the minimum side = img_size,
then cropped to img_size x img_size image.
* `[--hole_min_w]`: The minimum width of a hole (default: 48).
* `[--hole_max_w]`: The max width of a hole (default: 48).
* `[--hole_min_h]`: The minimum height of a hole (default: 96).
* `[--hole_max_h]`: The max height of a hole (default: 96).

**Example**: If you'd like to make a inference with a input image `***/GLCIC-pytorch/input.jpg` and create an output image `***/GLCIC-pytorch/output.jpg`, run the following command.

```bash
# in ***/GLCIC-pytorch/
$ python predict.py results/result/phase_*/model_cn_step* results/result/config.json input.jpg output.jpg
```
