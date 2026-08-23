

# AM-OIQA

# The source code of AM-OIQA.

**Omnidirectional Image Quality Assessment via Autoregressive Modeling**

This repository contains the official implementation of our paper:
mnidirectional Image Quality Assessment via Autoregressive Modeling

# Usages

## 1. dataset preparation

The viewport extraction module is in "./data/transfer.py", use the following command to extract the viewports:

```bash
# extract the viewports
transfer.py --input_path="path/to/your/ERP/images"--output_path="output/path/of/viewports"
```

## 2. Model Training and evaluation

You can easily train the proposed network with following command:

```bash
# train and evaluate the proposed network
python train_all.py --database=CVIQ
```

We adopted the pretrained SwinV2-T as the backbone of our model, which can be downloaded from this [site](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth). Before training, place the weight file under the weight directory and rename it to swinv2_tiny_patch4_window8_256.pth.

## 3. Cross-database evaluation

You can evaluate the cross-database performance of network with following command:

```bash
python cross_train.py --trainbase=CVIQ
```

We support the following databases: CVIQ, OIQA, and OIQ10K databases for cross-database evaluation.

# Contact

lxliu@bit.edu.cn  
yqliu@bit.edu.cn
zyhu@bit.edu.cn
