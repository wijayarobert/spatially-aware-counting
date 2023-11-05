# Spatially-Aware Class-Agnostic Object Counting
## A Quick Overview

![enter image description here](https://i.ibb.co/7RLcYCL/arch-3.png%22%20alt=%22arch-3%22%20border=%220%22)
Abstract. 

## Installation
#### 1. Download Dataset
We use FSC-147 dataset. Please download the dataset here:
 - [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)
#### 2. Set up Conda environment
To properly set up an Anaconda environment for executing the training and inference processes of "model_name," please execute the following commands:
```
conda create --name "ourmodel"-environ python=3.7
conda activate countx-environ
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install scipy
pip install imgaug
git clone git@github.com:wijayarobert/"model_name".git
```

## Usage

**Training** 
python train.py --output_dir "./results" --img_dir "/dir/images_384_VarV2" --gt_dir "/dir/gt_density_map_adaptive_384_VarV2" --class_file "/dir/ImageClasses_FSC147.txt" --FSC147_anno_file "/dir/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "dir/Train_Test_Val_FSC_147.json" >>./training.log 2>&1 &

**Inference** (validation)
python test.py --data_split "val" --output_dir "./test" --resume "./results/checkpoint-1000.pth" --img_dir "/dir/images_384_VarV2" --FSC147_anno_file "/dir/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/dir/Train_Test_Val_FSC_147.json"

**Inference** (testing)
python test.py --data_split "test" --output_dir "./test" --resume "./results/checkpoint-1000.pth" --img_dir "/dir/images_384_VarV2" --FSC147_anno_file "/dir/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/dir/Train_Test_Val_FSC_147.json"

## Model Checkpoint

The model checkpoint can be downloaded here: Google drive

## Results

Quantitative Results and Qualitative Results

## Acknowledgement
[niki-amini-naieni/CounTX](https://github.com/niki-amini-naieni/CounTX),    [Verg-Avesta/CounTR](https://github.com/Verg-Avesta/CounTR),  [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

