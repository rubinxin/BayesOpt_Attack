# BayesOpt Attack

This is the code repository for the query-efficient black-box attack method proposed in our paper [BayesOpt Adversarial Attack](https://openreview.net/pdf?id=Hkem-lrtvH). 

## Setup

Install the required libraries:
```
pip install -r requirements.txt 
```
To collect CIFAR10 data, run 
```
python setup_cifar.py
```
To download Inception-v3 model checkpoint, run 
```
python setup_inception.py
```
To collect ImageNet test data, check instructions [here](https://github.com/nesl/adversarial_genattack) 


## Usage:

Run BayesOpt Attack experiments: `python run_bayesopt_attack.py` followed by the following important arguments:
* `-f`  Attack target model/dataset: `mnist`, `cifar10` or `imagenet`
* `-m`  BayesOpt attack method: `GP` for GP-BO, `ADDGPLD` for ADDGP-BO or `GPLDR` for GP-BO-auto-dr
* `-ld`  Reduced dimension. `=196` for MNIST and CIFAR10, `=2304` for ImageNet. Not applicable when use GP-BO-auto-dr as it learns the reduced dimension automatically
* `-rd`  Upsampling method. Default `=BILI` for bilinear resizing
* `-nsub`  Number of subspaces from decomposing the image dimensions. Only applicable when use ADDGP-BO: `=14` for MNIST, `=12` for CIFAR10 or `=27` for ImageNet.

Other arguments are described in `run_bayesopt_attack.py`.

## Examples
  Case 1: GP-BO attack on trained MNIST classifier for 900 iterations with reduced dimension of 14x14x1. Attack 1 correctly classified image with ID=0 on the other 9 class labels: 
  ```
  python run_bayesopt_attack.py -f='mnist' -m='GP' -nitr=900 -rd='BILI' -ld=196 -i=0 -ntg=9
  ```
 
  Case 2: GP-BO-auto-dr attack on trained CIFAR10 classifier for 900 iterations. Attack 1 correctly classified image with ID=0  on the other 9 class labels: 
  ```
  python run_bayesopt_attack.py -f='cifar10' -m='GPLDR' -nitr=900 -i=0 -ntg=9
  ```
  Case 3: ADDGP-BO attack on trained ImageNet classifier for 2000 iterations with reduced dimension of 48x48x1 and 27 subspaces. Attack 1 correctly classified image with ID=0  on a random class label: 
  ```
  python run_bayesopt_attack.py -f='imagenet' -m='ADDGPLD' -nitr=2000 -rd='BILI' -ld=2304 -nsub=27 -i=0 -ntg=1 
  ```

## Citation
Please cite our paper if you would like to use the code.

```
@inproceedings{
Ru2020BayesOpt,
title={BayesOpt Adversarial Attack},
author={Binxin Ru and Adam Cobb and Arno Blaas and Yarin Gal},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Hkem-lrtvH}
}
```