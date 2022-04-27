# Feature Base Model Train

## Train Model
- gtea

```bash
# single gpu
export CUDA_VISIBLE_DEVICES=1
python main.py  --validate -c config/gtea/sliding_mstcn_split1.yaml --seed 0
python main.py  --validate -c config/gtea/sliding_mstcn_split2.yaml --seed 0
python main.py  --validate -c config/gtea/sliding_mstcn_split3.yaml --seed 0
python main.py  --validate -c config/gtea/sliding_mstcn_split4.yaml --seed 0

# multi gpu
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 main.py --launcher pytorch --validate -c config/gtea/sliding_mstcn_split1.yaml --seed 0
```

## Test Model
- gtea

```bash
python main.py  --test -c config/gtea/etesvs_split1.yaml --weights=output/ETESVS_gtea_split1/ETESVS_gtea_split1_best.pkl
```