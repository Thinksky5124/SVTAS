# Usage
## Pre-train weight
- You can find some pre-train weight in [mmaction2](https://github.com/open-mmlab/mmaction2).

## Usage
- There are some `.sh` example files in `script` dictory. 

### Train

Switch `--validate` on to validating while training.

- multi-gpus train
```bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/launch.py \
    --m train \
    --launcher pytorch \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
```
- single-gpu train
```bash
export CUDA_VISIBLE_DEVICES=0

python tools/launch.py \
    --mode train \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
```

Indicating `-c` to set configuration, and one can flexible add `-o` in the script to update it.

```bash
python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    tools/launch.py \
    --m train \
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

Indicate `-o resume_epoch` to resume, It will training from ```resume_epoch``` epoch, SVTAS will auto load optimizers parameters and checkpoints from `./output` folder, as it is the default output destination.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    tools/launch.py \
    --m train \
    --launcher pytorch \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
    -o resume_epoch=5

```

### Finetune

Indicate `--weights` to load pretrained parameters, SVTAS will auto treat it as a finetune mission.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    tools/launch.py \
    --m train \
    --launcher pytorch \
    --validate \
    -c CONFIG_PATH \
    --seed SEED
    --weights=./outputs/example/path_to_weights
```

Note: SVTAS will NOT load shape unmatched parameters.

### Test

Switch `--test` on to start test mode, and indicate `--weights` to load pretrained model.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --nproc_per_node=4 \
    tools/launch.py \
    --m test \
    --launcher pytorch \
    --test \
    -c CONFIG_PATH \
    example.py \
    -c ./configs/example.yaml \
    --weights=./output/example/path_to_weights
```

# Visualization
```bash
# gtea
python tools/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 128
# 50salads
python tools/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 600
```