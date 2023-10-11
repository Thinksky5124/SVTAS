# **Important!**
**Warning**
- This repo **main** branch are under development so it will have much bugs, because it doesn't test completely!

**Note**
- If you want to reproduce paper list, please checkout branch to **svtas-paper**!

# Paper List
- Streaming Video Temporal Action Segmentation In Real Time, [![](https://img.shields.io/badge/arViv-@doi:2209.13808-orange.svg)](https://arxiv.org/pdf/2209.13808.pdf), **statu**: under review 

## Abstract

Temporal action segmentation (TAS) is a critical step toward long-term video understanding. Recent studies follow a pattern that builds models based on features instead of raw video picture information. However, we claim those models are trained complicatedly and limit application scenarios. It is hard for them to segment human actions of video in real time because they must work after the full video features are extracted. As the real-time action segmentation task is different from TAS task, we define it as streaming video real-time temporal action segmentation (SVTAS) task.

<div align="center">
  <img src="docs/image/demo.gif" width=440/></div>

# Framework Feature
- [x] Distribution train
- [x] Tensorboard visualization
- [x] Caculate model Params and Flops
- [x] Apex accelerate
- [x] Apex ditributedd accelerate
- [x] Pillow-SMID accelerate sample
- [x] Onnxruntime Infer Suppport
- [x] Support CAM Visualization
- [x] Assemble the `pytest` testing framework
- [x] Pytorch Profiler Suppport
- [ ] Pulg-in DeepSpeed
- [ ] Pulg-in Tritron

# Envirnment Prepare

- Linux Ubuntu 22.04+
- Python 3.10+
- PyTorch 2.1.0+
- CUDA 12.2+ 
- Pillow-SIMD (optional): Install it by the following scripts.
- FFmpeg 4.3.1+ (optional): For extract flow and visualize video cam

```bash
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```

- use pip to install environment

```bash
conda create -n torch python=3.10
python -m pip install --upgrade pip
pip install -r requirements/requirements_base.txt
```
- If report `correlation_cuda package no found`, you should read [Install](svtas/model/backbones/utils/liteflownet_v3/README.md)
- If you want to extract montion vector and residual image to video, you should install ffmpeg, for example, in ubuntu `sudo apt install ffmpeg`

# Document Dictionary
- [Prepare Datset](docs/prepare_dataset.md)
- [Usage](docs/usage.md)
- [Model Zoo](docs/model_zoo.md)
- [Tools Usage](docs/tools_usage.md)
- [Infer Guideline](docs/infer_guideline.md)
- [Add Test Case Guideline](docs/add_testcase_guideline.md)

# Citation
```bib
@misc{2209.13808,
Author = {Wujun Wen and Yunheng Li and Zhuben Dong and Lin Feng and Wanxiao Yang and Shenlan Liu},
Title = {Streaming Video Temporal Action Segmentation In Real Time},
Year = {2022},
Eprint = {arXiv:2209.13808},
}
```
# Acknowledgement
This repo borrowed code from many great open source libraries, thanks again for their selfless dedication.
- [mmaction2](https://github.com/open-mmlab/mmaction2)
- [paddlevideo](https://github.com/PaddlePaddle/PaddleVideo)
- [SlowFast](https://github.com/facebookresearch/SlowFast)