# RGB Base Model Train

## Prepare Pretrain Weight

- step 1 Down resnet pretrain weight checkpoint file from `./model/backbone.py`'s `model_urls`

- step 2 Move this checkpoint file in `./data` directory

## Train Model
- gtea

```bash
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
```
- egtea

```bash
# single gpu
export CUDA_VISIBLE_DEVICES=2
python main.py  --validate -c config/egtea/etesvs_split1.yaml --seed 0
python main.py  --validate -c config/egtea/etesvs_split2.yaml --seed 0
python main.py  --validate -c config/egtea/etesvs_split3.yaml --seed 0

python main.py  --validate -c config/egtea/etesvs_mobinetv2_split1.yaml --seed 0
# multi gpu
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/egtea/etesvs_mobinetv2_split1.yaml --seed 0
```
- 50salads

```bash
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
```
- breakfast

```bash
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
```
- thumos14

```bash
export CUDA_VISIBLE_DEVICES=3
python main.py  --validate -c config/thumos14/etesvs.yaml  --seed 0

# multi gpu
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/thumos14/etesvs_mobinetv2_split1.yaml --seed 0
```
## Test Model
- gtea

```bash
python main.py  --test -c config/gtea/etesvs_split1.yaml --weights=output/ETESVS_gtea_split1/ETESVS_gtea_split1_best.pkl
python main.py  --test -c config/gtea/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_MobileNetV2_gtea_split1/ETESVS_MobileNetV2_gtea_split1_best.pkl
```
- egtea

```bash
python main.py  --test -c config/egtea/etesvs_split1.yaml --weights=output/ETESVS_egtea_split1/ETESVS_egtea_split1_best.pkl
python main.py  --test -c config/egtea/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_MobileNetV2_egtea_split1/ETESVS_MobileNetV2_egtea_split1_epoch_00001.pkl
```

- 50salads

```bash
python main.py  --test -c config/50salads/etesvs_split1.yaml --weights=output/ETESVS_50salads_split1/ETESVS_50salads_split1_best.pkl
python main.py  --test -c config/50salads/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_MobileNetV2_50salads_split1/ETESVS_MobileNetV2_50salads_split1_best.pkl
python main.py  --test -c config/50salads/etesvs_mobinetv2_split1.yaml --weights=output/baseline/50salads_split1_baseline/ETESVS_MobileNetV2_50salads_split1_best.pkl

export CUDA_VISIBLE_DEVICES=2,3
export DECORD_EOF_RETRY_MAX=20480
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --test -c config/50salads/etesvs_split1.yaml --weights=output/ETESVS_50salads_split1/ETESVS_50salads_split1_best.pkl
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --test -c config/50salads/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_50salads_split1/ETESVS_50salads_split1_best.pkl
```
- thumos14

```bash
python main.py  --test -c config/thumos14/etesvs_mobinetv2_split1.yaml --weights=output/ETESVS_thumos14/ETESVS_thumos14_best.pkl
```