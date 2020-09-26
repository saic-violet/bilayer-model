# Fast Bi-layer Neural Synthesis of One-Shot Realistic Head Avatars

A project on the speed up of one-shot adversarially trained human pose to image translation models for mobile devices.

<img src="https://saic-violet.github.io/bilayer-model/assets/teaser.png"/>

## Installation

* Python 3.7
* Pytorch 1.3 or higher
* Apex (is required only for training, needs to be built from https://github.com/NVIDIA/apex)
* Face-alignment (https://github.com/1adrianb/face-alignment)
* Other packages are in requirements.txt
* Download pretrained_weights and runs from https://drive.google.com/drive/folders/11SwIYnk3KY61d8qa17Nlb0BN9j57B3L6

## Inference API usage

```python
import argparse
from infer import InferenceWrapper

args_dict = {
    'project_dir': '.',
    'init_experiment_dir': './runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}

# Initialization
module = InferenceWrapper(args_dict)

# Input data for intiialization and inference
data_dict = {
    'source_imgs': ..., # Size: H x W x 3, type: NumPy RGB uint8 image
    'target_imgs': ..., # Size: NUM_FRAMES x H x W x 3, type: NumPy RGB uint8 images
}

# Inference
data_dict = module(data_dict)

# Outputs (images are in [-1, 1] range, segmentation masks -- in [0, 1])
imgs = data_dict['pred_enh_target_imgs']
segs = data_dict['pred_target_segs']
```

For a concrete inference example, please refer to examples/inference.ipynb.

## Training

The example training scripts are in the scripts folder. The base model is trained first, the texture enhancer is trained afterwards. In order to reproduce the results from the paper, 8 GPUs with at least 24 GBs of memory are required, since batch normalization layers may be sensitive to the batch size.

## Datasets

Supported datasets should have the same structure as VoxCeleb2 (http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset:

```DATA_ROOT/[imgs, keypoints, segs]/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM[.jpg, .npy, .png]```

Please refer to the link above for more details.

Additionally, all training data must be annotated with keypoints obtained using face-alignment (or any other keypoints detection) library before training. Annotation with segmentation masks is optional, yet it significantly improves the performance of the method.

## Links

- Project page: https://saic-violet.github.io/bilayer-model
- ArXiv: https://arxiv.org/abs/2008.10174
- YouTube: https://youtu.be/54tji11VhOI

## Citation
```
@InProceedings{Zakharov20,
  author={Zakharov, Egor and Ivakhnenko, Aleksei and Shysheya, Aliaksandra and Lempitsky, Victor},
  title={Fast Bi-layer Neural Synthesis of One-Shot Realistic Head Avatars},
  booktitle = {European Conference of Computer vision (ECCV)},
  month = {August},
  year = {2020}}
```
