# ETESVS
End to End Stream Video Segmentation Network for Action Segmentation and Action Localization——You do not need to extract feature

## Abstract

Temporal action segmentation and localization is a challenge task which attracts many researchers’ attention recently. As a downstream tasks of action recognition, most studies focus on how to classify frames or regression boundary base on the video feature extracted by action recognition model. However, we claim that above approaches are two stage or three stage, which must train split two or three models, and hard to segment or localize video on real time, because previous model must work on the whole video feature extracted by action recognition model. In this paper, we introduce an end-to-end approach, which uses sliding windows method to classify every frame and end to end segment videos that means need to use action recognition model to extract feature. Our approach can deal with stream video and reduce the number of parameters by 10% and the number of calculation by 20% compared I3D with MS-TCN.

# Framework Feature
- [x] Distribution train
- [x] Tensorboard visualization
- [x] Caculate model Params and Flops
- [x] Apex accelerate
- [x] Apex ditributedd accelerate
- [x] Pillow-SMID accelerate sample

# Envirnment Prepare

- Linux Ubuntu 20.04+
- Python 3.8+
- PyTorch 1.11+
- CUDA 11.3+
- Cudnn 8.2+
- Pillow-SIMD (optional): Install it by the following scripts.

```bash
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```

- use pip to install environment

```bash
conda create -n torch python=3.8
python -m pip install --upgrade pip
pip install -r requirements.txt

# export
pip freeze > requirements.txt
```

# Baseline
- FSF:  Flops of Single Frame(G)
- FMSF: FLOPs of Model Single Forward(G)

## other utils model
| Model         | Param(M)  | FSF(G)        | FMSF(G)     | RES   | FRAMES | FPS |
| -----         | -----     |   -----       | -----       | ----- | -----  | --- |
| bmn           | 13.095319 | 28.6235698    | -           | 224   |  1x15  | -   |
| two_stream-i3d| 24.575126 | 4.078423      | 261.01912   | 224   |  1x64  | -   |
| ssd           | 12.298539 | 27.832209856  | -           | 224   |  1x15  | -   |
| asrf          | 1.3       | 0.01283328    | 1.283328    | 100   |  1x100 | -   |
| mstcn         | 0.796780  | 0.00791359944 | 0.791359944 | 100   |  1x100 | -   |
| tsm           | 24.380752 | 4.0625        | 65          | 224   |  1x16  | -   |
| asformer      | 1.13086   | -             | -           | 100   |  1x100 | -   |
| mobiV2+ms     | 15.518241 | 0.01609396    | 0.402349    | 224   |  1x30  | -   |

## gtea

| Model         | Param(M)  | FSF(G)        | FMSF(G)     | RES   | FRAMES | FPS | AUC      | F1@0.10   | F1@0.25   | F1@0.5    | Acc      | Edit     | mAP@0.5  | avgmAP   | pre-train    | split-train |
| -----         | -----     |   -----       | -----       | ----- | -----  | --- | -----    | -----     | -----     | -----     | -----    | -----    | -----    |  -----   | -----        | -----       |
| i3d+asformer  | 25.705986 | -             | -           | 224   |  1x64  | -   | -        | 90.1000%  | 88.8000%  | 79.2000%  | 79.7000% | 84.6000% | -        | -        | Kinetics-400 | yes         |
| i3d+asrf      | 25.875126 | 4.09125628    | 262.302448  | 224   |  1x64  | -   | -        | 89.4000%  | 87.8000%  | 79.8000%  | 77.3000% | 83.7000% | -        | -        | Kinetics-400 | yes         |
| i3d+mstcn     | 25.371906 | 4.086323      | 261.8104799 | 224   |  1x64  | -   | 82.9200% | 87.5000%  | 85.4000%  | 74.6000%  | 79.2000% | 81.4000% | 64.4500% | -        | Kinetics-400 | yes         |
| mobiV2+ms     | 15.518241 | 0.01609396    | 0.402349    | 224   |  1x30  | -   | 76.5571% | 82.7839%  | 79.1209%  | 68.8645%  | 76.3886% | 75.4410% | 58.7359% | 32.7357% | ImageNet1000 | no          |

## 50salads

| Model         | Param(M)  | FSF(G)        | FMSF(G)     | RES   | FRAMES | FPS | AUC      | F1@0.10   | F1@0.25   | F1@0.5    | Acc      | Edit     | mAP@0.5  | avgmAP   | pre-train    | split-train |
| -----         | -----     |   -----       | -----       | ----- | -----  | --- | -----    | -----     | -----     | -----     | -----    | -----    | -----    |  -----   | -----        | -----       |
| i3d+asformer  | 25.705986 | -             | -           | 224   |  1x64  | -   | -        | 85.1000%  | 83.4000%  | 76.0000%  | 85.6000% | 79.6000% | -        | -        | Kinetics-400 | yes         |
| i3d+asrf      | 25.875126 | 4.09125628    | 262.302448  | 224   |  1x64  | -   | -        | 84.9000%  | 83.5000%  | 77.3000%  | 84.5000% | 79.3000% | -        | -        | Kinetics-400 | yes         |
| i3d+mstcn     | 25.371906 | 4.086323      | 261.8104799 | 224   |  1x64  | -   | -        | 76.3000%  | 74.0000%  | 64.5000%  | 80.7000% | 67.9000% | -        | -        | Kinetics-400 | yes         |
| mobiV2+ms     | 15.518241 | 0.01609396    | 0.402349    | 224   |  1x30  | -   | 77.9101% | 66.1818%  | 62.9091%  | 56.3636%  | 81.1014% | 56.2990% | 67.7964% | 42.2856% | ImageNe1000  | no          |
| tsm+mstcn     | 25.177532 | 4.8784962     | -           | 224   |  1x30  | -   | 76.2602% | 67.1560%  | 65.3211%  | 55.4128%  | 84.2123& | 57.7032% | 66.9675% | 44.8097% | ImageNe1000  | no          |


## breakfast

| Model         | Param(M)  | FSF(G)        | FMSF(G)     | RES   | FRAMES | FPS | AUC      | F1@0.10   | F1@0.25   | F1@0.5    | Acc      | Edit     | mAP@0.5  | avgmAP   | pre-train    | split-train |
| -----         | -----     |   -----       | -----       | ----- | -----  | --- | -----    | -----     | -----     | -----     | -----    | -----    | -----    |  -----   | -----        | -----       |
| i3d+asformer  | 25.705986 | -             | -           | 224   |  1x64  | -   | -        | 76.0000%  | 70.6000%  | 57.4000%  | 73.5000% | 75.0000% | -        | -        | Kinetics-400 | yes         |
| i3d+asrf      | 25.875126 | 4.09125628    | 262.302448  | 224   |  1x64  | -   | -        | 75.3000%  | 68.9000%  | 56.1000%  | 67.6000% | 72.4000% | -        | -        | Kinetics-400 | yes         |
| i3d+mstcn     | 25.371906 | 4.086323      | 261.8104799 | 224   |  1x64  | -   | -        | 52.6000%  | 48.1000%  | 37.9000%  | 66.3000% | 61.7000% | -        | -        | Kinetics-400 | yes         |
| i3d+ms++      | -         | 0.01609396    | 0.402349    | 224   |  1x64  | -   | -        | 63.3000%  | 57.7000%  | 44.5000%  | 67.3000% | 64.5000% | -        | -        | Kinetics-400 | yes         |
| mobiV2+ms     | 15.518241 | 0.402349      | -           | 224   |  1x30  | -   | -        | -         | -         | -         | -        | -        | -        | -        | ImageNet1000 | no          |

# Prepare Data

Read Doc [Prepare Datset](doc/prepare_dataset.md)

# Model Train

## RGB Base model train

Read Doc [RGB Base Model Train](doc/rgb_base.md)

## Feature Base model train

Read Doc [Feature Base Model Train](doc/feature_base.md)

# Visualization
```bash
# gtea
python tools/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 120
# 50salads
python tools/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 600
# thumos14
python tools/convert_pred2img.py output/results/pred_gt_list data/thumos14/mapping.txt output/results/imgs --sliding_windows 256
```