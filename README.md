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

# Baseline and Benckmark
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

Read Doc [Bneckmark](doc/benckmark.md)

# Prepare Data

Read Doc [Prepare Datset](doc/prepare_dataset.md)

# Model Train and Test

## Usage

### Train

Switch `--validate` on to validating while training.

- multi-gpus train
```bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --launcher pytorch \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
```
- single-gpu train
```bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
```

Indicating `-c` to set configuration, and one can flexible add `-o` in the script to update it.

```bash
python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --launcher pytorch \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
    -o DATASET.batch_size=BATCH_SIZE 
```
Indicating `-o DATASET.batch_size=BATCH_SIZE` can update batch size to BATCH_SIZE.

After starting training, log files will generated, and its format is shown as below, it will output to both the screen and files. Default destination of log is under the `.log/` folder, and stored in the files named like `worker.0`, `worker.1` ...

[train phase] current time, current epoch/ total epoch, batch id, metrics, elapse time, ips, etc.:

    [12/28 17:31:26] epoch:[ 1/80 ] train step:0   loss: 0.04656 lr: 0.000100 top1: 1.00000 top5: 1.00000 elapse: 0.326 reader: 0.001s ips: 98.22489 instance/sec.

[eval phase] current time, current epoch/ total epoch, batch id, metrics, elapse time, ips, etc.:


    [12/28 17:31:32] epoch:[ 80/80 ] val step:0    loss: 0.20538 top1: 0.88281 top5: 0.99219 elapse: 1.589 reader: 0.000s ips: 20.14003 instance/sec.


[epoch end] current time, metrics, elapse time, ips, etc.

    [12/28 17:31:38] END epoch:80  val loss_avg: 0.52208 top1_avg: 0.84398 top5_avg: 0.97393 elapse_avg: 0.234 reader_avg: 0.000 elapse_sum: 7.021s ips: 136.73686 instance/sec.

[the best Acc]  

    [12/28 17:28:42] Already save the best model (top1 acc)0.8494

### Resume

Indicate `-o resume_epoch` to resume, It will training from ```resume_epoch``` epoch, ETESVS will auto load optimizers parameters and checkpoints from `./output` folder, as it is the default output destination.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --launcher pytorch \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
    -o resume_epoch=5

```

### Finetune

Indicate `--weights` to load pretrained parameters, ETESVS will auto treat it as a finetune mission.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --launcher pytorch \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
    --weights=./outputs/example/path_to_weights
```

Note: ETESVS will NOT load shape unmatched parameters.

### Test

Switch `--test` on to start test mode, and indicate `--weights` to load pretrained model.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --launcher pytorch \
    --test \
    -c CONFIG_PATH \
    main.py \
    -c ./configs/example.yaml \
    --weights=./output/example/path_to_weights
```

## RGB Base model train and test

Read Doc [RGB Base Model Train](doc/rgb_base.md)

## Feature Base model train and test

Read Doc [Feature Base Model Train](doc/feature_base.md)

# Visualization
```bash
# gtea
python tools/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 128
# 50salads
python tools/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 600
# breakfast
python tools/convert_pred2img.py output/results/pred_gt_list data/breakfast/mapping.txt output/results/imgs --sliding_windows 128
# thumos14
python tools/convert_pred2img.py output/results/pred_gt_list data/thumos14/mapping.txt output/results/imgs --sliding_windows 256
```