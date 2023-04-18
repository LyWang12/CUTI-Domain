# Model Barrier: A Compact Un-Transferable Isolation Domain for Model Intellectual Property Protection

Code release for "Model Barrier: A Compact Un-Transferable Isolation Domain for Model Intellectual Property Protection" (CVPR 2023)

## Paper

<div align=center><img src="./Figures/1.pdf" width="100%"></div>

[Model Barrier: A Compact Un-Transferable Isolation Domain for Model Intellectual Property Protection](https://arxiv.org/abs/2303.11078) 
(CVPR 2023)

We propose a Compact Un-Transferable Isolation Domain (CUTI-domain), which acts as a barrier to block illegal transfers from authorized to unauthorized domains, to protect the intellectual property (IP) of AI models.

## Prerequisites
The code is implemented with **CUDA 11.4**, **Python 3.8.5** and **Pytorch 1.8.0**.

## Datasets

### MNIST
MNIST dataset can be found [here](http://yann.lecun.com/exdb/mnist/).

### USPS
USPS dataset can be found [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2).

### SVHN
SVHN dataset can be found [here](http://ufldl.stanford.edu/housenumbers/train_32x32.mat).

### MNIST-M
MNIST-M dataset can be found [here](https://arxiv.org/pdf/1505.07818v4.pdf).

### CIFAR-10
CIFAR-10 dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

### STL-10
CIFAR-10 dataset can be found [here](https://opendatalab.com/STL-10).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### VisDA 2017

VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public).

## Running the code

Target-Specified CUTI-Domain
```
python train_ts_dight.py
```

Ownership Verification
```
python train_owner_dight.py
```

Target-free CUTI-Domain
```
python train_tf_dight.py
```

Applicability Authorization
```
python train_author_dight.py
```

## Citation
If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2303.11078):
```
@article{wang2023model,
  title={Model Barrier: A Compact Un-Transferable Isolation Domain for Model Intellectual Property Protection},
  author={Wang, Lianyu and Wang, Meng and Zhang, Daoqiang and Fu, Huazhu},
  journal={arXiv preprint arXiv:2303.11078},
  year={2023}
}
```

## Acknowledgements
Some codes are adapted from [NTL](https://github.com/conditionWang/NTL) and 
[SWIN-Transformer](https://github.com/microsoft/Swin-Transformer). We thank them for their excellent projects.

## Contact
If you have any problem about our code, feel free to contact
- lywang12@126.com
- wangmeng9218@126.com

or describe your problem in Issues.
