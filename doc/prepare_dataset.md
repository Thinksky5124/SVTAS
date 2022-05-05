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

To download I3D feature from [ms-tcn](https://github.com/yabufarha/ms-tcn) repo.

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

## Extract Optical Flow(Optional)
```bash
python tools/extract_flow.py -c config/extract_flow/extract_optical_flow.yaml -o data/gtea
```

## Dataset Normalization
```bash
# count mean and std from video
# gtea
python tools/transform_segmentation_label.py data/gtea data/gtea/groundTruth data/gtea --mode localization --fps 15
python tools/prepare_video_recognition_data.py data/gtea/label.json data/gtea/Videos data/gtea --negative_sample_num 100 --only_norm True --fps 15 --dataset_type gtea_rgb
python tools/prepare_video_recognition_data.py data/gtea/label.json data/gtea/flow data/gtea --negative_sample_num 100 --only_norm True --fps 15 --dataset_type gtea_flow

# egtea
python tools/prepare_video_recognition_data.py data/egtea/egtea.json data/egtea/Videos data/egtea --negative_sample_num 1000 --only_norm True --fps 24 --dataset_type egtea_rgb

# 50salads
python tools/transform_segmentation_label.py data/50salads data/50salads/groundTruth data/50salads --mode localization --fps 30
python tools/prepare_video_recognition_data.py data/50salads/label.json data/50salads/Videos data/50salads --negative_sample_num 1000 --only_norm True --fps 30 --dataset_type 50salads_rgb

# breakfast
python tools/transform_segmentation_label.py data/breakfast data/breakfast/groundTruth data/breakfast --mode localization --fps 15
python tools/prepare_video_recognition_data.py data/breakfast/label.json data/breakfast/Videos data/breakfast --negative_sample_num 10000 --only_norm True --fps 15 --dataset_type breakfast_rgb

# thumos14
python tools/prepare_video_recognition_data.py data/thumos14/gt.json data/thumos14/Videos data/thumos14 --negative_sample_num 1000 --only_norm True --fps 30 --dataset_type thumos14_rgb
```

Here releases dataset mean and std config

- gtea:
```txt
# rgb
mean RGB :[0.5505552534004328, 0.42423616561376576, 0.17930791124574694]
std RGB :[0.13311456349527262, 0.14092562889239943, 0.12356268405634434]
# flos
mean RGB :[0.9686297051020777, 0.9706158002294017, 0.972493270804535]
std RGB :[0.039060756165796726, 0.03689212641350189, 0.03209093941013171]
```
- egtea:
```txt
mean RGB ∶[0.47882690412518875, 0.30667687330914223, 0.1764174579795214]
std RGB :[0.26380785444954574, 0.20396220265286277, 0.16305419562005563]
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
- thumos14:
```txt
mean RGB ∶[0.384953972862144, 0.38326867429930167, 0.3525199505706894]
std RGB :[0.258450710004705, 0.2544892750057763, 0.24812118173426492]
```

## Convert Localization Label to Segmentation Label
```bash
# thumos14
python tools/transform_segmentation_label.py data/thumos14/gt.json data/thumos14/Videos data/thumos14 --mode segmentation --fps 30

# egtea
python tools/transform_egtea_label.py data/egtea/splits_label data/egtea/verb_idx.txt data/egtea
python tools/transform_segmentation_label.py data/egtea/egtea.json data/egtea/Videos data/egtea --mode segmentation --fps 24
```