# GLCIC-PyTorch

## About this repository

In this repository, we provide a pytorch-based implementation of [GLCIC](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) introduced by Iizuka et. al.

**4/9/2019: We finally succeeded to reproduce the paper result on CelebA dataset, and updated the pretrained model to the new one. Thanks for your long patience and please enjoy.**

![glcic](https://i.imgur.com/KY26J85.png)  
![result_1](https://i.imgur.com/LTYCUup.jpg)
![result_2](https://i.imgur.com/RR7MhNS.jpg)
![result_3](https://i.imgur.com/xOrTR4n.jpg)

## Dependencies

We tested our scripts in the following environment.

* Python: 3.6.5
* torch: 1.0.1.post2
* torchvision: 0.2.2.post3
* tqdm: 4.31.1
* Pillow: 5.4.1
* opencv-python: 4.0.0.21 
* numpy: 1.16.2
* scipy: 1.2.1
* GPU: Geforce GTX 1080Ti (12GB RAM) X 4

If you'd like to run our training script (i.e., train.py), we recommend you to
use a more than middle-range GPU such as GTX 1070(Ti) or GTX 1080(Ti).

## DEMO (Inference)

### 1. Download our pretrained generator and its training config file.
* Pretrained generator (Completion Network): [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EZmwLEHNXDNDkAzM7BFOpRABIbuW8iakXEiSWfPBx-4NQA?e=bzej00)
* Pretrained discriminator (Context Discriminator): [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EdEPzkhWS0xAhq0p_xON7tIBiX_HKviUAgvU-rLtF6uo8w?e=da8QDm) (optional)
* Training config file: [download](https://keiojp0-my.sharepoint.com/:u:/g/personal/snake_istobelieve_keio_jp/EWGLy7HYSfxNusolMnlFdPYBQ-osGwfNexe87AjmwSwlQQ?e=t5g7w3)

Both of the generator and the discriminator were trained on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
You don't have to download the discriminator since only a generator is needed to perform image completion (a discriminator is used only during training).
The training config file stores training settings in json format.

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

In this section, we introduce how to train a glcic model using CelebA dataset.

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
All the training images are stored in `img_align_celeba/train/`, while
the other test images are stored in `img_align_celeba/test/`.

### 2. Training

Just run the following command.

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ python train.py datasets/img_align_celeba results/demo/
```

Training results (trained models and inference results at each snapshot period) are to be stored in `results/demo/`.

The training procedure consists of the following three phases.  
1. In phase 1, only Completion Network (i.e., generator) is trained.
2. In phase 2, only Context Discriminator (i.e., discriminator) is trained, while Completion Network is frozen.
3. In phase 3, Both of the networksd are jointly trained.

In default settings, the numbers of training steps during phase 1, phase 2, and phase 3 are 90,000, 10,000, and 400,000, respectively.
Each snapshot period is set to 10,000, 2,000, and 10,000.
Bach size is 16.
Size of an input image is 160 x 160 (all input images are rescalled so that the minumum side is 160, then randomly cropped to 160 x 160 images).

Basically, the default hyper-parameters we use in `train.py` are set to the same settings as those used in the original paper **except for batch size due to lack of GPU memories**.
Please run the training script with batch size == 96 if you would like to reproduce the paper result (the pretrained model now we are sharing is trained with batch size == 96).
You can train a model with larger batch size by enabling `[--data_parallel]` flag (see [here](#data_parallel)) or tuning `[--bsize]` if you have some gpus and enough size of GPU memories.

## How to train with your own dataset ?

### 1. Prepare dataset

First you have to make a dataset directory in the following format.

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
images in `dataset/test/` are used for test inpainting at each
snapshot period.

### 2. Training

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ mv dataset/ datasets/
$ python train.py datasets/dataset/ results/result/ [--data_parallel] [--cn_input_size] [--ld_input_size] [--steps_1] [--steps_2] [--steps_3] [--snaperiod_1] [--snaperiod_2] [--snaperiod_3] [--bsize] [--bdivs] [--optimizer]
```

Results for each training phase (trained models and test inpainting results) are to be stored in `results/result/`.

<a name="data_parallel"></a>
**Arguments**  
* `<dataset>` (required): Path to the dataset directory.
* `<result>` (required): Path to the result directory.
* `[--data_parallel]`: Suppose you have *N* gpus available, the whole model is copied to the *N* gpus and
each minibatch of batch size (*bsize* / *N*) is sent to each gpu. Then, all the grads are reduced and
the updated parameters are broadcasted to all the gpus. The whole procedure is iterated until all the three training phases end.
You can speed up the training procedure ideally up to *N* times faster. (default: False (store true)).
* `[--cn_input_size]`: Input size of Completion Network. All input images are rescalled so that the minimum side = cn\_input\_size
then randomly cropped to cn\_input\_size x cn\_input\_size images (default: 160).
* `[--ld_input_size]`: Input size of Local Discriminator (default: 96).
* `[--init_model_cn]`: Path to a pretrained model of Completion Network. It is used as the initial parameters (default: None).
* `[--init_model_cd]`: Path to the pretrained model of Context Discriminator (default: None).
* `[--steps_1]`: The number of training steps during phase 1 (default: 90,000).
* `[--steps_2]`: The number of training steps during phase 2 (default: 10,000).
* `[--steps_3]`: The number of training steps during phase 3 (default: 400,000).
* `[--snaperiod_1]`: Snapshot period during phase 1 (default: 10,000).
* `[--snaperiod_2]`: Snapshot period during phase 2 (default: 2,000).
* `[--snaperiod_3]`: Snapshot period during phase 3 (default: 10,000).
* `[--max_holes]`: The max number of holes to be randomly generated (default: 1).
* `[--hole_min_w]`: The minimum width of a hole (default: 48).
* `[--hole_max_w]`: The max width of a hole (default: 96).
* `[--hole_min_h]`: The minimum height of a hole (default: 48).
* `[--hole_max_h]`: The max height of a hole (default: 96).
* `[--bsize]`: Batch size (default: 16).
* `[--bdivs]`: Devide a single training step of batch size *bsize* into *bdivs* steps of batch size *bsize / bdivs* on a single gpu and
update the model parameters every *bdivs* steps. It produce the same training results as when *bdivs == 1* with smaller gpu memory consumption.
However, the whole training procedure would somehow slow down due to the spliting.
If you enable `[--data_parallel]` option and *N* gpus are available, each minibach of batch size *bsize / N* is
sent to each gpu. Then a single training step of batch size *bsize / N* on a single gpu is devided into *bdivs* steps of batch size *(bsize / N) / bdivs*,
and the parameters are updated every *bdivs* steps. (default: 1).
* `[--optimizer]`: 'adadelta' or 'adam' (default: 'adadelta').

**Example**: If you'd like to train a model with batch size 24 with data_parallel option and the other parameters are default values, run the following command.

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
* `[--img_size]`: Input size of Completion Network. Input images are rescalled so that the minimum side = img\_size
then randomly cropped to img\_size x img\_size images (default: 160).
* `[--max_holes]`: The max number of holes to be randomly generated (default: 5).
* `[--hole_min_w]`: The minimum width of a hole (default: 24).
* `[--hole_max_w]`: The max width of a hole (default: 48).
* `[--hole_min_h]`: The minimum height of a hole (default: 24).
* `[--hole_max_h]`: The max height of a hole (default: 48).

**Example**: If you'd like to make a inference with an input image `{path_to_this_repo}/GLCIC-PyTorch/input.jpg` and create an output image `{path_to_this_repo}/GLCIC-PyTorch/output.jpg`, run the following command.

```bash
# in {path_to_this_repo}/GLCIC-PyTorch/,
$ python predict.py results/result/phase_3/model_cn_step{step_number} results/result/config.json input.jpg output.jpg
```
