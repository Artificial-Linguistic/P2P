# Fine Tuning Pre-Trained Image Models for Point Cloud Analysis with Point to Pixel Promting

Created by Lavanya Joshi
 
![image](https://github.com/Artificial-Linguistic/P2P/assets/79073015/7f799a16-2c28-499b-a4e0-b29dae5c9b4a)

The Point-to-Pixel (P2P) prompting framework is an advanced and integrated approach designed to achieve high-quality 3D reconstructions by converting 3D point clouds into 2D images while maintaining geometric fidelity. This framework incorporates various state-of-the-art techniques and models, optimizing each stage of the transformation process to ensure both accuracy and visual quality

## Preparation

### Installation Pre-Requisites

Python 3.10.12
CUDA 12.2
PyTorch
timm 
torch_scatter
PointNet++

```
Installing PointNet++

cd lib
pip install -r requirements.txt

```

# Data Preparation

Download the Processed Dataset from [[Google Drive]](https://drive.google.com/drive/folders/1kR2QILZOq1PhCyMMyGwMCvf5ZJbt4hhk?usp=sharing).
Or you can download the offical ModelNet from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip), and process it by yourself.

The data is expected to be in the following file structure:
```
    P2P/
    |-- config/
    |-- data/
        |-- ModelNet40/
            |-- modelnet40_shape_names.txt
            |-- modelnet_train.txt
            |-- modelnet_test.txt
            |-- modelnet40_train_8192pts_fps.dat
            |-- modelnet40_test_8192pts_fps.dat
    |-- dataset/
```



