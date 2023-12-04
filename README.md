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

### :key: LV/RV Segmentation

## Acknowledgement
This code is extended from the following repositories.
- [ngp_pl](https://github.com/kwea123/ngp_pl)

We thank the authors for releasing their code. Please also consider citing their work.