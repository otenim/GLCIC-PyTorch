# GLCIC-pytorch


## About this repository

Here, we provide a pytorch implementation of [GLCIC](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) introduced by Iizuka et. al.

![glcic](https://i.imgur.com/KY26J85.png)  
![result_1](https://i.imgur.com/SYkn6Uo.png)  
![result_2](https://i.imgur.com/T8GGx1g.jpg)  

## Dependencies

We tested our scripts in the following environment.

* Python: 3.5, 3.6
* torch: 0.4.1, 1.0.0
* torchvision: 0.2.1
* tqdm: 4.24.0
* Pillow: 5.2.0
* numpy: <font color="Red">**1.14.5 (important)**</font>
* pyamg: <font color="Red">**3.3.2 (important)**</font>
* scipy: 1.1.0
* GPU: Geforce GTX 1080Ti (12GB RAM) X 1

If you would like to run our training script, we recommend you to
use a more than middle-range GPU such as GTX 1070 or GTX 1080(Ti).

## DEMO (Inference)

### 1. Download our pretrained model and the training config file.

Pretrained model: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/ETadvo335qNJvYa-chglWFsBmkVM9xH_c9WkdNkMBpVo2Q?e=H9g60Q)  
Training config file: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EWQL72AQjUhNsdsVN-AtAIMBWR_Xx-dBr48gbqpR47mlgQ?e=eCDjcl)

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
# in {path_to_this_repo}/GLCIC-pytorch/,
$ python predict.py model_cn_step400000 config.json images/test_1.jpg out.jpg
```

## DEMO (Training)

Here, we introduce how to train a model using CelebA dataset.

### 1. Download the dataset

First, download the dataset (i.e., img\_align\_celeba.zip) from [this official link](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM).

Second, run the following commands.

```bash
$ unzip img_align_celeba.zip
$ mv img_align_celeba/ {path_to_this_repo}/GLCIC-pytorch/datasets/
$ cd {path_to_this_repo}/GLCIC-pytorch/datasets/
$ python make_dataset.py img_align_celeba/
```

Originally, all the images are stored in `img_align_celeba/`,
and the last command splits the dataset into two subsets; training dataset (80%) and test dataset (20%). All the training images are stored in `img_align_celeba/train/`, while
the other images are in `img_align_celeba/test/`.


### 2. Training

Just run the following command.

```bash
# in {path_to_this_repo}/GLCIC-pytorch/,
$ python train.py datasets/img_align_celeba results/demo/
```

Training results (trained models and inference results at each snapshot period) are to be saved in `results/demo/`.

The training procedure consists the following three phases.  
1. In phase 1, only Completion Network (i.e., generator) is trained.
2. In phase 2, only Context Discriminator (i.e., discriminator) is trained, while Completion Network is frozen.
3. In phase 3, Both of the Completion Network and the Context Discriminator are jointly trained.

Under default settings, the numbers of training steps during phase 1, phase 2, and phase 3 are 90,000, 10,000, and 400,000. Each snapshot period is set to 18,000, 2,000, and 80,000, respectively. Bach size is set to 16, while input image size is 160 x 160 (all the input images are rescalled so that the minumum side is 160, then randomly cropped to 160 x 160 images).

Basically, hyper parameters and the model architecture (except for batch size) is exactly the same as described in the paper.

## How to train with your own dataset ?

### 1. Prepare dataset

To train a model using your own dataset, first you have to make a dataset
directory in the following format.

```
dataset/ # the directory name can be anything.
    |____train/
    |       |____XXXX.jpg # png images are also accepted.
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
# in {path_to_this_repo}/GLCIC-pytorch/,
$ mv dataset/ datasets/
$ python train.py datasets/dataset/ results/result/ [--data_parallel] [--cn_input_size] [--ld_input_size] [--steps_1] [--steps_2] [--steps_3] [--snaperiod_1] [--snaperiod_2] [--snaperiod_3] [--bsize] [--bdivs] [--optimizer]
```

Training results for each training phase (trained models and test completion results at each snapshot period) are to be stored in `results/result/`.

**Arguments**  
* `<dataset>` (required): Path to the dataset directory.
* `<result>` (required): Path to the result directory.
* `[--data_parallel]`: Suppose you have *N* gpus available, the whole model is copied to the *N* gpus and
each minibatch of batch size (*bsize* / *N*) is sent to each gpu. Then, all the grads are reduced and
the updated parameters are broadcasted to all the gpus. The procedure is looped until the training ends.
You can speed up the training procedure ideally up to *N* times faster. (default: False (store true)).
* `[--cn_input_size]`: Input size of Completion Network (default: 160). All the input images are rescalled so that the length of the minimum side = cn\_input\_size,
then randomly cropped to cn\_input\_size x cn\_input\_size images.
* `[--ld_input_size]`: Input size of Local Discriminator (default: 96).
* `[--init_model_cn]`: Path to the pre-trained model of Completion Network. This is used as the initial parameters (default: None).
* `[--init_model_cd]`: Path to the pre-trained model of Context Discriminator (default: None).
* `[--steps_1]`: Training iterations in phase 1 (default: 90,000).
* `[--steps_2]`: Training iterations in phase 2 (default: 10,000).
* `[--steps_3]`: Training iterations in the last phase (default: 400,000).
* `[--snaperiod_1]`: Snapshot period in phase 1 (default: 18,000).
* `[--snaperiod_2]`: Snapshot period in phase 2 (default: 2,000).
* `[--snaperiod_3]`: Snapshot period in the last phase (default: 80,000).
* `[--max_holes]`: The max number of holes (default: 1).
* `[--hole_min_w]`: The minimum width of a hole (default: 48).
* `[--hole_max_w]`: The max width of a hole (default: 48).
* `[--hole_min_h]`: The minimum height of a hole (default: 96).
* `[--hole_max_h]`: The max height of a hole (default: 96).
* `[--bsize]`: Batch size (default: 16).
* `[--bdivs]`: Devide a single training step of batch size *bsize* into *bdivs* steps of batch size *bsize / bdivs* on the same gpu, and
update the parameters every *bdivs* steps. You can get the same training results as when *bdivs == 1* with smaller gpu memory.
However, the training procedure would somehow slow down due to the spliting.
If you have enabled [--data_parallel] option and *N* gpus are available, each minibach of batch size *bsize / N* is
sent to each gpu, then a single training step of batch size *bsize / N* on one gpu is devided into *bdivs* steps of batch size *(bsize / N) / bdivs*,
and the parameters are updated every *bdivs* steps. (default: 1).
* `[--optimizer]`: 'adadelta' or 'adam' (default: 'adadelta').

**Example**: If you'd like to train a model with batch size 24 with data_parallel option, and the other parameters are default values, run the following command.

```bash
# in {path_to_this_repo}/GLCIC-pytorch/,
$ python train.py datasets/dataset results/result --data_parallel --bsize 24
```

## How to infer with your own dataset ?

Suppose you've finished train a model and the result directory is `{path_to_this_repo}/GLCIC-pytorch/results/result`, run the following command.

```bash
# in {path_to_this_repo}/GLCIC-pytorch/,
$ python predict.py results/result/phase_3/model_cn_step{step_number} results/result/config.json <input_img> <output_img> [--max_holes] [--img_size] [--hole_min_w] [--hole_max_w] [--hole_min_h] [--hole_max_h]
```

**Arguments**  
* `<input_img>` (required): Path to an input image.
* `<output_img>` (required): Path to an output image.
* `[--img_size]`: Input size of Completion Network (default: 160). The input image are rescalled so that the length of the minimum side = img\_size,
then randomly cropped to a img\_size x img\_size image.
* `[--max_holes]`: The max number of holes (default: 1).
* `[--hole_min_w]`: The minimum width of a hole (default: 48).
* `[--hole_max_w]`: The max width of a hole (default: 48).
* `[--hole_min_h]`: The minimum height of a hole (default: 96).
* `[--hole_max_h]`: The max height of a hole (default: 96).

**Example**: If you'd like to make a inference with a input image `{path_to_this_repo}/GLCIC-pytorch/input.jpg` and create an output image `{path_to_this_repo}/GLCIC-pytorch/output.jpg`, run the following command.

```bash
# in {path_to_this_repo}/GLCIC-pytorch/,
$ python predict.py results/result/phase_3/model_cn_step{step_number} results/result/config.json input.jpg output.jpg
```

## Future Work

* **Currently, we are trying to reproduce the paper results using exactly the same model settings including batch size.
We plan to share the pretrained model and the weights if the experiments succeeded, so please wait little bit more !**
* Adding data-parallel training option.
* Adding some kind of an option to train models with large batch size.
