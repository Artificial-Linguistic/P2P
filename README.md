# Fine Tuning Pre-Trained Image Models for Point Cloud Analysis with Point to Pixel Prompting

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

### Data Preparation

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
## Usage

### Train

```
bash tool/train.sh EXP_NAME CONFIG_PATH DATASET
```

For example, to train P2P model with ConvNeXt-B-1k as base model on the ModelNet40 dataset:

```
bash tool/train.sh p2p_ConvNeXt-B-1k config/ModelNet40/p2p_ConvNeXt-B-1k.yaml ModelNet40
```

### Test

```
bash tool/test.sh EXP_NAME CONFIG_PATH DATASET
```

For example, to test P2P model with ConvNeXt-B-1k as base model the ModelNet40 dataset:

```
bash tool/test.sh p2p_ConvNeXt-B-1k config/ModelNet40/p2p_ConvNeXt-B-1k.yaml ModelNet40
```

### Reproduce

```
bash tool/reproduce.sh DATASET MODEL
```

For example, to reproduce results of P2P model with ConvNeXt-B-1k as base model on the ModelNet40 dataset with our provided checkpoint:

```
bash tool/reproduce.sh ModelNet40 ConvNeXt-B-1k
```

You can download the already pre-trained Conv-NeXt Models from [[Google Drive]](https://drive.google.com/drive/folders/1gglAunXt55tbJlszvkJm9OYpovk5uBca?usp=drive_link)

The Pre-Trained Weights are need to in path:
```
P2P/
    |-- pretrained/
        |-- reproduce/
            |-- ckpt/
                |-- ModelNet40/
                    |-- ConvNeXt-B-1k-ModelNet40.pth
                |-- ScanObjectNN/
                    |-- ConvNeXt-B-1k-ScanObjectNN.pth
```


