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
* pyamg: 3.3.2
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

The pretrained model was trained with [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.  
Hyper parameters and the model architecture is exactly the same as what described in the paper.  
The training config file saves training settings in json format.


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

Here, we introduce how to train the model using CelebA dataset.

### 1. Download the dataset

First, download the dataset (i.e., img_align_celeba.zip) from [this official link](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM).

Second, run the following commands.

```bash
$ unzip img_align_celeba.zip
$ mv img_align_celeba ***/GLCIC-pytorch/datasets/
$ cd ***/GLCIC-pytorch/datasets/
$ python make_dataset.py img_align_celeba
```

Originally, all the images are stored in `img_align_celeba/`,
and the last command splits the dataset into two subsets; training dataset and test dataset. All the training images are stored in `img_align_celeba/train`, while
the other images are in `img_align_celeba/test`.


### 2. Training

Just run the following command.

```
# in ***/GLCIC-pytorch/
$ python train.py datasets/img_align_celeba results/test
```

Training results (trained models and inference results at each snapshot period) are to be saved in `results/test`.

The training procedure consists the following three phases.  
1. In phase 1, only Completion Network is trained.
2. In phase 2, only Contect Discriminator is trained (Completion Network is frozen).
3. In phase 3, the Completion Network and the Context Discriminator are jointly trained.

Under the default settings, the numbers of training iterations of phase 1, phase 2, and phase 3 are 90,000, 10,000, and 400,000, and each snapshot period is set to 18,000, 2,000, and 80,000, respectively. Bach size is set to 16, while the input size is 160 x 160 (all the input images are rescalled so that the length of the minumum side is 160 pixel, then randomly cropped to 160 x 160 images).
