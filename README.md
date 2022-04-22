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

## Download Dataset

prepare data follow below instruction.
- data directory file tree
```txt
─── data
    ├── 50salads
    ├── breakfast
    ├── gtea
    └── ...
```

### gtea and 50salads and breakfast

The video action segmentation model uses [breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/), [50salads](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/) and [gtea](https://cbs.ic.gatech.edu/fpv/) data sets.

- Dataset tree example
```txt
─── gtea
    ├── Videos
    │   ├── S1_Cheese_C1.mp4
    │   ├── S1_Coffee_C1.mp4
    │   ├── S1_CofHoney_C1.mp4
    │   └── ...
    ├── groundTruth
    │   ├── S1_Cheese_C1.txt
    │   ├── S1_Coffee_C1.txt
    │   ├── S1_CofHoney_C1.txt
    │   └── ...
    ├── splits
    │   ├── test.split1.bundle
    │   ├── test.split2.bundle
    │   ├── test.split3.bundle
    │   └── ...
    └── mapping.txt
```

### thumos14
[Thumos14](http://crcv.ucf.edu/THUMOS14/home.html) dataset is temporal action localization dataset.
- Dataset tree
```txt
─── thumos14
    ├── Videos
    │   ├── video_test_0000896.mp4
    │   ├── video_test_0000897.mp4
    │   ├── video_validation_0000482.mp4
    │   └── ...
    ├── groundTruth
    │   ├── video_test_0000897.txt
    │   ├── video_test_0000897.txt
    │   ├── video_validation_0000482.txt
    │   └── ...
    ├── val_list.txt
    ├── test_list.txt
    └── mapping.txt
```

## Dataset Normalization
```bash
# count mean and std from video
# gtea
python utils/transform_segmentation_label.py data/gtea data/gtea/groundTruth data/gtea --mode localization --fps 15
python utils/prepare_video_recognition_data.py data/gtea/label.json data/gtea/Videos data/gtea --negative_sample_num 100 --only_norm True --fps 15 --dataset_type gtea

# 50salads
python utils/transform_segmentation_label.py data/50salads data/50salads/groundTruth data/50salads --mode localization --fps 30
python utils/prepare_video_recognition_data.py data/50salads/label.json data/50salads/Videos data/50salads --negative_sample_num 1000 --only_norm True --fps 30 --dataset_type 50salads

# breakfast
python utils/transform_segmentation_label.py data/breakfast data/breakfast/groundTruth data/breakfast --mode localization --fps 15
python utils/prepare_video_recognition_data.py data/breakfast/label.json data/breakfast/Videos data/breakfast --negative_sample_num 10000 --only_norm True --fps 15 --dataset_type breakfast

# thumos14
python utils/prepare_video_recognition_data.py data/thumos14/gt.json data/thumos14/Videos data/thumos14 --negative_sample_num 1000 --only_norm True --fps 30 --dataset_type thumos14
```

Here releases dataset mean and std config

- gtea:
```txt
mean RGB :[0.5505552534004328, 0.42423616561376576, 0.17930791124574694]
std RGB : [0.13311456349527262, 0.14092562889239943, 0.12356268405634434]
```
- 50salads:
```txt
mean RGB ∶[0.5139909998345553, 0.5117725498677757，0.4798814301515671]
std RGB :[0.23608918491478523, 0.23385714300069754, 0.23755006337414028]
```
- breakfast:
```txt
mean RGB ∶[0.4245283568405083, 0.3904851168609079, 0.33709139617292494]
std RGB :[0.26207845745959846, 0.26008439810422, 0.24623600365905168]
```
- thumos14
```txt
mean RGB ∶[0.384953972862144, 0.38326867429930167, 0.3525199505706894]
std RGB :[0.258450710004705, 0.2544892750057763, 0.24812118173426492]
```

## Convert Localization Label to Segmentation Label
```bash
# thumos14
python utils/transform_segmentation_label.py data/thumos14/gt.json data/thumos14/Videos data/thumos14 --mode segmentation --fps 30

# egtea
python utils/transform_egtea_label.py data/egtea/splits_label data/egtea/verb_idx.txt data/egtea
python utils/transform_segmentation_label.py data/egtea/egtea.json data/egtea/Videos data/egtea --mode segmentation --fps 24
```

# Prepare Pretrain Weight

- step 1 Down resnet pretrain weight checkpoint file from `./model/backbone.py`'s `model_urls`

- step 2 Move this checkpoint file in `./data` directory

# Train Model
```bash
# gtea
# single gpu
export CUDA_VISIBLE_DEVICES=1
python main.py  --validate -c config/gtea/etesvs_split1.yaml --seed 0
python main.py  --validate -c config/gtea/etesvs_split2.yaml --seed 0
python main.py  --validate -c config/gtea/etesvs_split3.yaml --seed 0
python main.py  --validate -c config/gtea/etesvs_split4.yaml --seed 0

python main.py  --validate -c config/gtea/etesvs_mobinetv2_split1.yaml --seed 0
# multi gpu
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/gtea/etesvs_split1.yaml --seed 0

# 50salads
export CUDA_VISIBLE_DEVICES=1
export DECORD_EOF_RETRY_MAX=20480
python main.py  --validate -c config/50salads/etesvs_split1.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split2.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split3.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split4.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split5.yaml --seed 0

python main.py  --validate -c config/50salads/etesvs_mobinetv2_split1.yaml --seed 0

# multi gpu
export CUDA_VISIBLE_DEVICES=0,1
export DECORD_EOF_RETRY_MAX=20480
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/50salads/etesvs_split1.yaml --seed 0
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/50salads/etesvs_mobinetv2_split1.yaml --seed 0

# breakfast
export CUDA_VISIBLE_DEVICES=2
export DECORD_EOF_RETRY_MAX=20480
python main.py  --validate -c config/breakfast/etesvs_split1.yaml  --seed 0
python main.py  --validate -c config/breakfast/etesvs_split2.yaml  --seed 0
python main.py  --validate -c config/breakfast/etesvs_split3.yaml  --seed 0
python main.py  --validate -c config/breakfast/etesvs_split4.yaml  --seed 0

python main.py  --validate -c config/breakfast/etesvs_mobinetv2_split1.yaml --seed 0

# multi gpu
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/breakfast/etesvs_split1.yaml --seed 0
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/breakfast/etesvs_mobinetv2_split1.yaml --seed 0

# thumos14
export CUDA_VISIBLE_DEVICES=3
python main.py  --validate -c config/thumos14/etesvs.yaml  --seed 0

# multi gpu
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/thumos14/etesvs_mobinetv2_split1.yaml --seed 0


# egtea
# single gpu
export CUDA_VISIBLE_DEVICES=2
python main.py  --validate -c config/egtea/etesvs_split1.yaml --seed 0
python main.py  --validate -c config/egtea/etesvs_split2.yaml --seed 0
python main.py  --validate -c config/egtea/etesvs_split3.yaml --seed 0
```
# Test Model
```bash
# gtea
python main.py  --test -c config/gtea/etesvs_split1.yaml --weights=output/ETESVS_gtea_split1/ETESVS_gtea_split1_best.pkl
python main.py  --test -c config/gtea/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_MobileNetV2_gtea_split1/ETESVS_MobileNetV2_gtea_split1_best.pkl

# 50salads
python main.py  --test -c config/50salads/etesvs_split1.yaml --weights=output/ETESVS_50salads_split1/ETESVS_50salads_split1_best.pkl
python main.py  --test -c config/50salads/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_MobileNetV2_50salads_split1/ETESVS_MobileNetV2_50salads_split1_best.pkl
python main.py  --test -c config/50salads/etesvs_mobinetv2_split1.yaml --weights=output/baseline/50salads_split1_baseline/ETESVS_MobileNetV2_50salads_split1_best.pkl

export CUDA_VISIBLE_DEVICES=2,3
export DECORD_EOF_RETRY_MAX=20480
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --test -c config/50salads/etesvs_split1.yaml --weights=output/ETESVS_50salads_split1/ETESVS_50salads_split1_best.pkl
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --test -c config/50salads/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_50salads_split1/ETESVS_50salads_split1_best.pkl

```

# Visualization
```bash
# gtea
python utils/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 120
# 50salads
python utils/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 600
```