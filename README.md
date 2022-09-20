# Personalized Human Action Recognition (PHAR) with Learning by Asking (LBA)

## Installation 

The scripts are tested on Windows 10 and Anaconda Python 3.6.  
You also need to install the following modules.   
  
```
$ pip install tqdm simplejson blitz-bayesian-pytorch matplotlib opencv-python  
$ conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch  
$ pip install pandas==0.25.0  
$ pip install scikit-learn==0.22.1
```
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 9.0 installation


## How to use

1. Run ```python extractAIR.py```.
1. Run ```python extractNTU.py```.
1. Run ```python label.py```
1. Run ```python train.py```.
1. Run ```python test.py```
1. Run ```python retrain_by_file.py```
1. Run ```python retrain_by_data.py```