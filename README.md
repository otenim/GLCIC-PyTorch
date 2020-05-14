# GLCIC-PyTorch

## About this repository
This repository provides a pytorch-based implementation of [GLCIC](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) introduced by Iizuka et. al.

**4/9/2019: We finally succeeded in reproducing the paper result on CelebA dataset and uploaded the new pretrained model. Thanks for your long patience and please enjoy it.**

![glcic](https://i.imgur.com/KY26J85.png)  
![result_1](https://i.imgur.com/LTYCUup.jpg)
![result_2](https://i.imgur.com/RR7MhNS.jpg)
![result_3](https://i.imgur.com/xOrTR4n.jpg)

## Dependencies

The scripts were tested in the following environment.

* Python: 3.6.5
* torch: 1.5.0 (cuda 10.0)
* torchvision: 0.6.0
* tqdm: 4.46.0
* Pillow: 7.1.2
* opencv-python: 4.2.0
* numpy: 1.18.4
* GPU: Geforce GTX 1080Ti (12GB RAM) X 4

A middle-class GPU (e.g., GTX1070(Ti) or GTX1080(Ti)) is required
to execute the training script (i.e., train.py).

## DEMO (Inference)

### 1. Download pretrained generator model and training config file.
* [Required] Pretrained generator model (Completion Network): [download (onedrive)](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EZmwLEHNXDNDkAzM7BFOpRABIbuW8iakXEiSWfPBx-4NQA?e=bzej00), [download (google drive)](https://drive.google.com/open?id=11hemcglluPPqG9rqQeoQGfZiw1P0LrxH)
* [Optional] Pretrained discriminator model (Context Discriminator): [download (onedrive)](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EdEPzkhWS0xAhq0p_xON7tIBiX_HKviUAgvU-rLtF6uo8w?e=da8QDm), [download (google drive)](https://drive.google.com/open?id=1NIVlvPidgpbCcu-HEbH-hnGYuJHlLgKT)
* [Required] Training config file: [download (onedrive)](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EWGLy7HYSfxNusolMnlFdPYBQ-osGwfNexe87AjmwSwlQQ?e=t5g7w3), [download (google drive)](https://drive.google.com/open?id=1HlVLaz-GLEzwIYnLrnW1v_F6vMAz0ycJ)

Both the generator and discriminator models were trained on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
Note that you don't need to get the dicriminator model because only generator is necessary
to perform image completion.
The training config files contains training settings in json format.

### 2. Inference

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ python predict.py model_cn config.json images/test_2.jpg test_2_out.jpg
```

Left: input image  
Center: input image of Completion Network  
Right: output image of Completion Network  

<img src="https://imgur.com/jXSYxeh.jpg" width="160px">
<img src="https://imgur.com/TM35BMq.jpg" width="160px">
<img src="https://imgur.com/2SXStFZ.jpg" width="160px">
<img src="https://imgur.com/EPaMwK9.jpg" width="160px">
<img src="https://imgur.com/zotuLM3.jpg" width="160px">

## DEMO (Training)

This section introduces how to train a glcic model using CelebA dataset.

### 1. Download the dataset

First, download the dataset (i.e., img\_align\_celeba.zip) from [this official link](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM).

Second, run the following commands.

```bash
$ unzip img_align_celeba.zip
$ mv img_align_celeba/ {path_to_this_repo}/GLCIC-PyTorch/datasets/
$ cd {path_to_this_repo}/GLCIC-PyTorch/datasets/
$ python make_dataset.py img_align_celeba/
```

The last command splits the dataset into two subsets: (1) a training dataset (80%) and (1) a test dataset (20%).
The training dataset is stored in `img_align_celeba/train/`,
while the test dataset is in `img_align_celeba/test/`.

### 2. Training

Just run the following command.

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ python train.py datasets/img_align_celeba results/demo/
```

Training results (trained model snapshots and inference results) are to be saved in `results/demo/`.

The training procedure consists of the following three phases.  
1. In phase 1, only Completion Network (i.e., generator) is trained.
2. In phase 2, only Context Discriminator (i.e., discriminator) is trained, while Completion Network is frozen.
3. In phase 3, Both of the networksd are jointly trained.

By default, the training steps during phase 1, 2, and 3 are set to 90,000, 10,000, and 400,000, respectively.
Input size is set to 160 x 160; all input images are rescaled so that the minimum edge becomes 160
then they are randomly cropped into 160 x 160 images. Batch size is set to 16.

The above settings are based on the original paper **except for batch size** due to lack of GPU memory space.
Please run train.py with batch size == 96 if you would like to reproduce the paper result (the pretrained model provided in this repository has been trained with batch size == 96).
If you would like to change batch size or enable multiple gpus, please see [here](#arguments)
(specifically, use `data_parallel` and `bsize` options).

## How to train with your own dataset ?

### 1. Prepare dataset

First you are required to make a dataset directory like below.

```
dataset/ # arbitrary name can be used.
    |____train/
    |       |____XXXX.jpg # .png format is also acceptable.
    |       |____OOOO.jpg
    |____test/
            |____oooo.jpg
            |____xxxx.jpg  
```

Images in `dataset/train/` are used to train models, while
images in `dataset/test/` are to perform image completion at each
snapshot period.

### 2. Training

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ mv dataset/ datasets/
$ python train.py datasets/dataset/ results/result/ [--data_parallel] [--cn_input_size] [--ld_input_size] [--steps_1] [--steps_2] [--steps_3] [--snaperiod_1] [--snaperiod_2] [--snaperiod_3] [--bsize] [--bdivs] [--optimizer]
```

Outputs of each training phase (trained model snapshots and test inpainting results) are to be saved in `results/result/`.

<a name="arguments"></a>
**Arguments**  
* `<dataset>` (required): Path to the dataset directory.
* `<result>` (required): Path to the result directory.
* `[--data_parallel]`: Suppose you have *N* gpus available, firstly the entire model is copied into *N* gpus.
Then, an input of batch size = *bsize* is divided into *N* fractions of batch size = *bsize*/*N*,
and each of them is transferred to a corresponding gpu.
After backpropagation on each gpu, all the updated parameters across *N* gpus are averaged,
then the output parameters are broadcasted to all the gpus.
This procedure is iterated until all the three training phases finish.
You can speed up the training procedure up to *N* times faster, ideally (default: False (store true)).
* `[--cn_input_size]`: Input size of Completion Network. All input images are rescalled so that the minimum side = `cn_input_size`,
then they are randomly cropped into `cn\_input\_size` x `cn\_input\_size` images (default: 160).
* `[--ld_input_size]`: Input size of Local Discriminator (default: 96).
* `[--init_model_cn]`: Path to a pretrained model of Completion Network.
The specified model is used as the initial parameters of Completion Network (default: None).
* `[--init_model_cd]`: Path to the pretrained model of Context Discriminator (default: None).
* `[--steps_1]`: Training steps during phase 1 (default: 90,000).
* `[--steps_2]`: Training steps during phase 2 (default: 10,000).
* `[--steps_3]`: Training steps during phase 3 (default: 400,000).
* `[--snaperiod_1]`: Snapshot period during phase 1 (default: 10,000).
* `[--snaperiod_2]`: Snapshot period during phase 2 (default: 2,000).
* `[--snaperiod_3]`: Snapshot period during phase 3 (default: 10,000).
* `[--max_holes]`: The maximum number of holes to be randomly generated and applied to each input image (default: 1).
* `[--hole_min_w]`: The minimum width of a hole (default: 48).
* `[--hole_max_w]`: The maximum width of a hole (default: 96).
* `[--hole_min_h]`: The minimum height of a hole (default: 48).
* `[--hole_max_h]`: The maximum height of a hole (default: 96).
* `[--bsize]`: Batch size (default: 16).
* `[--optimizer]`: 'adadelta' or 'adam' (default: 'adadelta').
* `[--bdivs]`: Devide a single training step of batch size *bsize* into *bdivs* steps of batch size *bsize*/*bdivs*, which produces the same training results as when `bdivs` = 1 but uses smaller gpu memory space at the cost of speed. This option can be used together with `data_parallel` (default: 1).

**Example**: If you'd like to train a model with batch size 24 with `data_parallel` option and leave the other settings as default, run the following command.

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ python train.py datasets/dataset results/result --data_parallel --bsize 24
```

## How to infer with your own dataset ?

Suppose you've finished train a model and the result directory is `{path_to_this_repo}/GLCIC-PyTorch/results/result`, run the following command.

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ python predict.py results/result/phase_3/model_cn_step{step_number} results/result/config.json <input_img> <output_img> [--max_holes] [--img_size] [--hole_min_w] [--hole_max_w] [--hole_min_h] [--hole_max_h]
```

**Arguments**  
* `<input_img>` (required): Path to an input image.
* `<output_img>` (required): Path to an output image.
* `[--img_size]`: Input size of Completion Network. Input images are rescalled so that the minimum side = img\_size then they are randomly cropped into `img\_size` x `img\_size` images (default: 160).
* `[--max_holes]`: The maximum number of holes to be randomly generated (default: 5).
* `[--hole_min_w]`: The minimum width of a hole (default: 24).
* `[--hole_max_w]`: The maximum width of a hole (default: 48).
* `[--hole_min_h]`: The minimum height of a hole (default: 24).
* `[--hole_max_h]`: The maximum height of a hole (default: 48).

**Example**: If you'd like to make an inference with an input image `{path_to_this_repo}/GLCIC-PyTorch/input.jpg` and create an output image `{path_to_this_repo}/GLCIC-PyTorch/output.jpg`, run the following command.

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ python predict.py results/result/phase_3/model_cn_step{step_number} results/result/config.json input.jpg output.jpg
```
