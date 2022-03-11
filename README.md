# ETETemporalSegmentation
End to End Video Temporal Segmention

# Envirnment Preapre
```bash
conda create --prefix=/mnt/wenwujun/torch python=3.8
python -m pip install --upgrade pip
pip install -r requirements.txt

# export
pip freeze > requirements.txt
```

# 测量baseline

## GTEA

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


# segmentation model train
```bash
# gtea
export CUDA_VISIBLE_DEVICES=2
python main.py  --validate -c applications/LightWeight/config/split/gtea/ms_tcn_GTEA.yaml --seed 0
python main.py  --validate -c applications/LightWeight/config/split/gtea/asrf_GTEA.yaml --seed 0

# 50salads
python main.py  --validate -c config/breakfast/ete_tsm_mstcn.yaml
```

# one-shot train

## prepare data
```bash
# gtea
python applications/LightWeight/prepare_ete_data_list.py \
                        --split_list_path data/gtea/splits \
                        --label_path data/gtea/groundTruth \
                        --output_path data/gtea/split_frames \
                        --window_size 60 \
                        --strike 15
```


## train model
```bash
# gtea
# single gpu
export CUDA_VISIBLE_DEVICES=1
python main.py  --validate -c applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml --seed 0
python main.py  --validate -c applications/LightWeight/config/one_shot/gtea/tsm_gtea_crop_train.yaml --seed 0
# multi gpu
export CUDA_VISIBLE_DEVICES=2,3
python -B -m paddle.distributed.launch --gpus="2,3"  --log_dir=./output main.py  --validate -c applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml --seed 0

# 50salads

# breakfast
python main.py  --validate -c applications/LightWeight/config/one_shot/breakfast/ete_tsm_mstcn.yaml --seed 0
```
## test model
```bash
python main.py  --test -c applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml --weights=./output/ETEMSTCN/ETEMSTCN_best.pdparams
```

# infer model
```bash
# export infer model
python tools/export_model.py -c applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml \
                                -p output/ETEMSTCN/ETEMSTCN_best.pdparams \
                                -o inference/ETEMSTCN

# use infer model to extract video feature
python3.7 tools/predict.py --input_file data/gtea/split_frames/test.split1.bundle \
                           --config applications/LightWeight/config/one_shot/gtea/ete_tsm_mstcn.yaml \
                           --model_file inference/ETEMSTCN/ETEMSTCN.pdmodel \
                           --params_file inference/ETEMSTCN/ETEMSTCN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False \
                           --batch_size 1
```