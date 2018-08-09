# GLCIC-pytorch


## About this repository

Here, we provide a high-quality pytorch implementation of [GLCIC](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) introduced by Iizuka et. al.

![network](https://i.imgur.com/wOnxWNc.png "Network")

## Dependencies

We tested our scripts in the following environment.

* Python: 3.5 or 3.6
* torch: 0.4.1 or 0.4.0
* torchvision: 0.2.1
* tqdm: 4.24.0
* Pillow: 5.2.0
* numpy: 1.15.0
* pyamg: 4.0.0
* scipy: 1.1.0
* GPU: Geforce GTX 1080Ti (12GB RAM) X 1

If you would like to run our training script, we recommend you to
use a more than middle-range GPU such as GTX 1070 or GTX 1080(Ti).

All the above dependent libraries can be installed with pip command.

## DEMO (Inference)

### 1. Download our pretrained model and the training config file.

**For Python 3.5 Users**
Pretrained model: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/Eaosyb919AJPjYau4ALWmKUB2i0L1lVh0dqVxhB2aHwBhg?e=GQBkP4)  
Training config file: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/Ebu3pP2wG2FKt1rRZzF_yEkBsVgxkBdJ28poeDfmGTz3aA?e=gpoAGc)

**For Python 3.6 Users**
Pretrained model: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EXPGbI_yvFNIhXNI7WgtgMkBdMbxJAdWJWbI5hNBJtHWUg?e=OYWDlH)  
Training config file: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/Ed0myTTrjN9FiX8sYwr4dsYBOEyj3pH_EQbu31HadjUvlw?e=DwAwaO)

The pretrained model was trained with CelebA dataset.  
The hyper parameters and the model architecture is exactly the same
as what described in the paper.

Training config file saves training settings in json format.  
This file is required to make models predict in the same environment as training time.

### 2. Inference

## DEMO (Training)
