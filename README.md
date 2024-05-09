# CardiacField: Computational Echocardiography for Universal Screening

This repository holds the Pytorch implementation of [CardiacField: Computational Echocardiography for Universal Screening](https://njuvision.github.io/CardiacField/). If you find our code useful in your research, please consider citing:

```
@article{shen2023cardiacfield,
  title={CardiacField: Computational Echocardiography for Universal Screening},
  author={Shen, Chengkang and Zhu, Hao and Zhou, You and Liu, Yu and Yi, Si and Dong, Lili and Zhao, Weipeng and Brady, David and Cao, Xun and Ma, Zhan and Lin, Yi},
  journal={Research Square},
  year={2023}
}
```

## Introduction
In this repository, we present an upgraded, high-speed version of our project, CardiacField. Below, you will find a comprehensive guide detailing the steps to effectively utilize our code.

## Quickstart
This repository has been developed using Python v3.8 and Pytorch v1.10.0 on Ubuntu 18.04. Our experiments were exclusively performed on an NVIDIA A100 GPU. For a complete list of dependencies, please refer to the [`requirements.txt`](requirements.txt). We recommend installing Python v3.8 from [Anaconda](https://www.anaconda.com/) and installing Pytorch (= 1.10.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. Please note, this implementation adheres to stringent requirements due to dependencies on certain libraries. Consequently, if installation issues arise due to hardware or software incompatibilities, we regrettably do not plan to extend support to other platforms, although contributions to address these issues are welcome.
### Hardware
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB, CUDA 11.3 (might work with older version)

### Software
* Clone this repo by `git clone https://github.com/cshennju/NeuralCMF.git`
* Python libraries
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`
* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

### :key: Training
For demonstrating the 3D heart reconstruction process, we have included a sample of 2D data along with its corresponding positional parameters in the [2d_example folder](2d_example) for demonstration. This data was acquired by rotating a 2DE probe 360 degrees around apex of the heart. We then apply our proprietary methods for estimating positional parameters to determine the exact position of each image within the 3D structure of the heart. The code for these positional parameter estimation methods will be made available shortly.

```
python train_heart.py --root_dir './2d_example/' --exp_name '/2d_example/'
```

### :key: Inference
After training, you can easily generate a 3D heart volume with the following command:
```
python vis_3D.py --root_dir 'example'
```
This command will produce a .mat file. To visualize the 3D heart model in applications like 3D Slicer or other compatible visualization software, you can convert the .mat file into a .nii.gz file using the command below:
```
python mat2nii.py
```
In our repository, we've included an example checkpoint file located in the [ckpts](folder). Additionally, we provide the reconstructed 3D heart model in two formats: example_vol.mat (.mat format) and example_vol.nii.gz (NIfTI format).

### :key: LV/RV Segmentation
In order to obtain the precise EDV and ESV to accurately calculate EF for cardiac function assessment, we first perform the uniform sampling on the reconstructed 3D heart to generate several 2D slices parallel to the apical four-chamber view, and then use the segmentation model developed  to automatically classify the area of the LV and RV. After the LV and RV segmentation, we calculate the volume of LV and RV according to the widely used Simpson’s rule in the clinical examinations. (The model has been updated and will be available soon.)

## Acknowledgement
This code is extended from the following repositories.
- [ngp_pl](https://github.com/kwea123/ngp_pl)

We thank the authors for releasing their code. Please also consider citing their work.