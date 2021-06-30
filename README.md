# GLCIC-PyTorch

This repository provides a pytorch-based implementation of [GLCIC](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) introduced by Iizuka et. al.

![glcic](https://i.imgur.com/KY26J85.png)  
![result_1](https://i.imgur.com/LTYCUup.jpg)
![result_2](https://i.imgur.com/RR7MhNS.jpg)
![result_3](https://i.imgur.com/xOrTR4n.jpg)

- [GLCIC-PyTorch](#glcic-pytorch)
  - [Requirements](#requirements)
  - [DEMO (Inference)](#demo-inference)
    - [1. Download pretrained generator model and training config file.](#1-download-pretrained-generator-model-and-training-config-file)
    - [2. Inference](#2-inference)
  - [DEMO (Training)](#demo-training)
    - [1. Download the dataset](#1-download-the-dataset)
    - [2. Training](#2-training)
  - [How to train with custom dataset ?](#how-to-train-with-custom-dataset-)
    - [1. Prepare dataset](#1-prepare-dataset)
    - [2. Training](#2-training-1)
  - [How to perform infenrece with custom dataset ?](#how-to-perform-infenrece-with-custom-dataset-)

## Requirements

Our scripts were tested in the following environment.

* Python: 3.7.6
* torch: 1.9.0 (cuda 11.1)
* torchvision: 0.10.0
* tqdm: 4.61.1
* Pillow: 8.2.0
* opencv-python: 4.5.2.54
* numpy: 1.19.2
* GPU: Geforce GTX 1080Ti (12GB RAM) X 4

You can install all the requirements by executing below.

```sh
# in <path-to-this-repo>/
pip install -r requirements.txt
```

## DEMO (Inference)

### 1. Download pretrained generator model and training config file.
* [Required] Pretrained generator model (Completion Network): [download (google drive)](https://drive.google.com/file/d/1hsi1Fy0ITiZYTsJ_De-nAVUciuJ8Bql9/view?usp=sharing)
* [Optional] Pretrained discriminator model (Context Discriminator): [download (google drive)](https://drive.google.com/file/d/1_jRuqirwOuiJCg1H73LSq1HuwyG2GON4/view?usp=sharing)
* [Required] Training config file: [download (google drive)](https://drive.google.com/file/d/1yGfQp8U5zcVRYOBxF3-VCZ8TnMAtWBsk/view?usp=sharing)

Both the generator and discriminator were trained on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
Note that you don't need to have dicriminator when performing image completion (discriminator is needed only during training).

### 2. Inference

```bash
# in <path-to-this-repo>/
python predict.py model_cn config.json images/test_2.jpg test_2_out.jpg
```

**Left**: raw input image  
**Center**: masked input image  
**Right**: inpainted output image    

<img src="https://imgur.com/jXSYxeh.jpg" width="200px">
<img src="https://imgur.com/TM35BMq.jpg" width="200px">
<img src="https://imgur.com/2SXStFZ.jpg" width="200px">
<img src="https://imgur.com/EPaMwK9.jpg" width="200px">
<img src="https://imgur.com/zotuLM3.jpg" width="200px">

## DEMO (Training)

This section introduces how to train a glcic model using CelebA dataset.

### 1. Download the dataset

Download the dataset from [this official link](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM).

Then, execute the following commands.

```bash
# unzip dataset
unzip img_align_celeba.zip
# move dataset
mv img_align_celeba/ <path-to-this-repo>/datasets/
# move into datasets/ directory
cd <path-to-this-repo>/datasets/
# make dataset
python make_dataset.py img_align_celeba/
```

The last command splits the dataset into training dataset (80%) and test dataset (20%) randomly.

### 2. Training

Run the following command.

```bash
# in <path-to-this-repo>
python train.py datasets/img_align_celeba results/demo/
```

Training results (model snapshots & test inpainted outputs) are to be saved in ``results/demo/``.

The training procedure consists of the following three phases.  
* **Phase 1**: trains only generator.
* **Phase 2**: trains only discriminator, while generator is frozen.
* **Phase 3**: both generator and discriminator are jointly trained.

Default settings of ``train.py`` are based on the original paper **except for batch size**.
If you need to reproduce the paper result, add ``--data_parallel --bsize 96`` when executing training.

## How to train with custom dataset ?

### 1. Prepare dataset

You have to prepare a dataset in the following format.

```
dataset/ # any name is OK
    |____train/ # used for training
    |       |____XXXX.jpg # .png format is also acceptable.
    |       |____OOOO.jpg
    |       |____....
    |____test/  # used for test
            |____oooo.jpg
            |____xxxx.jpg  
            |____....
```

Both `dataset/train` and `dataset/test` are required.

### 2. Training

```bash
# in <path-to-this-repo>/
# move dataset
mv dataset/ datasets/
# execute training
python train.py datasets/dataset/ results/result/ [--data_parallel (store true)] [--cn_input_size (int)] [--ld_input_size (int)] [--init_model_cn (str)] [--init_model_cd (str)] [--steps_1 (int)] [--steps_2 (int)] [--steps_3 (int)] [--snaperiod_1 (int)] [--snaperiod_2 (int)] [--snaperiod_3 (int)] [--bsize (int)] [--bdivs (int)]
```

<a name="arguments"></a>
**Arguments**  
* `<dataset>` (required): path to the dataset directory.
* `<result>` (required): path to the result directory.
* `[--data_parallel (store true)]`: when this flag is enabled, models are trained in data-parallel way. If *N* gpus are available, *N* gpus are used during training (default: disabled).
* `[--cn_input_size (int)]`: input size of generator (completion network). All input images are rescalled so that the minimum side is equal to `cn_input_size` then randomly cropped into `cn_input_size` x `cn_input_size` (default: 160).
* `[--ld_input_size (int)]`: input size of local discriminator (default: 96). Input size of global discriminator is the same as `[--cn_input_size]`.
* `[--init_model_cn (str)]`: path to a pretrained generator, used as its initial weights (default: None).
* `[--init_model_cd (str)]`: path to a pretrained discriminator, used as its initial weights (default: None).
* `[--steps_1 (int)]`: training steps during phase 1 (default: 90,000).
* `[--steps_2 (int)]`: training steps during phase 2 (default: 10,000).
* `[--steps_3 (int)]`: training steps during phase 3 (default: 400,000).
* `[--snaperiod_1 (int)]`: snapshot period during phase 1 (default: 10,000).
* `[--snaperiod_2 (int)]`: snapshot period during phase 2 (default: 2,000).
* `[--snaperiod_3 (int)]`: snapshot period during phase 3 (default: 10,000).
* `[--max_holes (int)]`: maximum number of holes randomly generated and applied to each input image (default: 1).
* `[--hole_min_w (int)]`: minimum width of a hole (default: 48).
* `[--hole_max_w (int)]`: maximum width of a hole (default: 96).
* `[--hole_min_h (int)]`: minimum height of a hole (default: 48).
* `[--hole_max_h (int)]`: maximum height of a hole (default: 96).
* `[--bsize (int)]`: batch size (default: 16). **bsize >= 96 is strongly recommended**.
* `[--bdivs (int)]`: divide a single training step of batch size = *bsize* into *bdivs* steps of batch size = *bsize*/*bdivs*, which produces the same training results as when `bdivs` = 1 but uses smaller gpu memory space at the cost of speed. This option can be used together with `data_parallel` (default: 1).

**Example**: If you train a model with batch size 24 with `data_parallel` option and leave the other settings as default, run the following command.

```bash
# in <path-to-this-repo>/
python train.py datasets/dataset results/result --data_parallel --bsize 24
```

## How to perform infenrece with custom dataset ?

Assume you've finished training and result directory is `<path-to-this-repo>/results/result`.

```bash
# in <path-to-this-repo>/
python predict.py results/result/phase_3/model_cn_step<step-number> results/result/config.json <input_img> <output_img> [--max_holes (int)] [--img_size (int)] [--hole_min_w (int)] [--hole_max_w (int)] [--hole_min_h (int)] [--hole_max_h (int)]
```

**Arguments**  
* `<input_img>` (required): path to an input image.
* `<output_img>` (required): path to an output image.
* `[--img_size (int)]`: input size of generator. Input images are rescalled so that the minimum side = `img_size` then randomly cropped into `img_size` x `img_size` (default: 160).
* `[--max_holes (int)]`: maximum number of holes to be randomly generated (default: 5).
* `[--hole_min_w (int)]`: minimum width of a hole (default: 24).
* `[--hole_max_w (int)]`: maximum width of a hole (default: 48).
* `[--hole_min_h (int)]`: minimum height of a hole (default: 24).
* `[--hole_max_h (int)]`: maximum height of a hole (default: 48).

**Example**: If you make an inference with an input image `<path-to-this-repo>/input.jpg` and save output image as `<path-to-this-repo>/output.jpg`, run the following command.

```bash
# in <path-to-this-repo>/
python predict.py results/result/phase_3/model_cn_step{step_number} results/result/config.json input.jpg output.jpg
```
