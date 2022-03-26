# ETESVS
End to End Stream Video Segmentation Network for Action Segmentation and Action Localization——You do not need to extract feature

## Abstract

Temporal action segmentation and localization is a challenge task which attracts many researchers’ attention recently. As a downstream tasks of action recognition, most studies focus on how to classify frames or regression boundary base on the video feature extracted by action recognition model. However, we claim that above approaches are two stage or three stage, which must train split two or three models, and hard to segment or localize video on real time, because previous model must work on the whole video feature extracted by action recognition model. In this paper, we introduce an end-to-end approach, which uses sliding windows method to classify every frame and end to end segment videos that means need to use action recognition model to extract feature. Our approach can deal with stream video and reduce the number of parameters by 10% and the number of calculation by 20% compared I3D with MS-TCN.

# Todo list
- [ ] distribution data parallel
- [ ] distribution change to torchrun

# Envirnment Prepare
```bash
conda create -n torch python=3.8
python -m pip install --upgrade pip
pip install -r requirements.txt

# export
pip freeze > requirements.txt
```

# Baseline

## gtea

| Model |   Param(M) | Flops(G) |   RES   |   FRAMES |  FPS |   AUC |   F1@0.5  |   mAP@0.5 |   Top1 Acc    |   pre-train  |    fine-tune   |   split-train |
| ----- |   -----   |   -----   |   -----   |   -----   |   -----   |   -----   |   ----- |   ----- |   ----- |   ----- |   ----- |   ----- |
| tsm |   24.380752 | 4.087136256 |   224   |   1x15  |  -   |   -  |   -  |   - |   98.86%  |  是  |   gtea    |
| i3d+mstcn |   13.095319 | 28.6235698 |   224   |   1x15  |  -   |   82.92%  |   74.6%  |   64.45% |   -  |  -  | -   |    yes   |
| tsm+mstcn |   25.177532 | 4.8784962 |   224   |   1x15  |  -   |   83.78%  |   91.99%  |   80.93% |   98.86%  |  ImageNet1000  |   gtea    |  yes |
| bmn |   13.095319 | 28.6235698 |   224   |   1x15  |  -   |   -  |   74.6%  |   - |   -  |  是  | -   |    -   |
| i3d |   12.298539 | 27.832209856 |   224   |   1x15  |  -   |   -  |   74.6%  |   - |   -  |  是  | -   |    -   |
| ssd |   12.298539 | 27.832209856 |   224   |   1x15  |  -   |   -  |   74.6%  |   - |   -  |  是  | -   |    -   |
| asrf |   1.3 | 1.283328 |   224   |   1x15  |  -   |   -  |   79.8%  |   - |   -  |  是  | -   |    -   |
| mstcn |   0.796780 | 0.791359944 |   224   |   1x15  |  -   |   -  |   79.8%  |   - |   -  |  是  | -   |    -   |

## 50salads
| Model |   Param(M) | Flops(G) |   RES   |   FRAMES |  FPS |   AUC |   F1@0.5  |   mAP@0.5 |   Top1 Acc    |   pre-train  |    fine-tune   |   split-train |
| ----- |   -----   |   -----   |   -----   |   -----   |   -----   |   -----   |   ----- |   ----- |   ----- |   ----- |   ----- |   ----- |
| tsm |   24.380752 | 4.087136256 |   224   |   1x15  |  -   |   -  |   -  |   - |   98.86%  |  是  |   gtea    |

# Prepare Data

## Download Dataset

prepare data follow below instruction.
- data tree
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
python utils/prepare_video_recognition_data.py data/thumos14/gt.json data/thumos14/Videos data/thumos14 --negative_sample_num 1000 --only_norm True --fps 30
```
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
mean RGB ∶[0.5139909998345553, 0.5117725498677757，0.4798814301515671]
std RGB :[0.23608918491478523, 0.23385714300069754, 0.23755006337414028]
```
- thumos14
```txt
mean RGB ∶[0.5139909998345553, 0.5117725498677757，0.4798814301515671]
std RGB :[0.23608918491478523, 0.23385714300069754, 0.23755006337414028]
```

## Convert Localization Label to Segmentation Label
```bash
# thumos14
python utils/transform_segmentation_label.py data/thumos14/gt.json data/thumos14/Videos data/thumos14 --mode segmentation --fps 30
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
# multi gpu
export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.launch --nproc_pemain.pyr_node=2  --launcher pytorch --validate -c config/gtea/ete_tsm_mstcn.yaml --seed 0

# 50salads
export CUDA_VISIBLE_DEVICES=3
export DECORD_EOF_RETRY_MAX=20480
python main.py  --validate -c config/50salads/etesvs_split1.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split2.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split3.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split4.yaml --seed 0
python main.py  --validate -c config/50salads/etesvs_split5.yaml --seed 0

python main.py  --validate -c config/50salads/etesvs_split1.yaml --seed 0 -o resume_epoch=7
# breakfast
python main.py  --validate -c config/breakfast/etesvs_split1.yaml  --seed 0
python main.py  --validate -c config/breakfast/etesvs_split2.yaml  --seed 0
python main.py  --validate -c config/breakfast/etesvs_split3.yaml  --seed 0
python main.py  --validate -c config/breakfast/etesvs_split4.yaml  --seed 0
```
# Test Model
```bash
# gtea
python main.py  --test -c config/gtea/etesvs_split1.yaml --weights=output/ETESVS_gtea_split1/ETESVS_gtea_split1_best.pkl
```

# Visualization
```bash
# gtea
python utils/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs
```