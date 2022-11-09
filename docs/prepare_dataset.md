# Prepare Data
## Download Dataset

prepare data follow below instruction.
- data directory file tree
```txt
─── data
    ├── 50salads
    ├── egtea
    ├── gtea
    └── ...
```

### gtea and 50salads and egtea

The video action segmentation model uses [egtea](https://cbs.ic.gatech.edu/fpv/), [50salads](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/) and [gtea](https://cbs.ic.gatech.edu/fpv/) data sets.

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

### thumos14 and TVSerious

You can download thumos14 dataset in [website](https://www.crcv.ucf.edu/THUMOS14/home.html) and TVSerious in [website](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html)
Thumos14 dataset is temporal action localization dataset.

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
python tools/extract/extract_flow.py -c config/extract_flow/extract_optical_flow_fastflownet.yaml -o data/gtea
python tools/extract/extract_flow.py -c config/extract_flow/extract_optical_flow_raft.yaml -o data/gtea
python tools/extract/extract_flow.py -c config/extract_flow/extract_optical_flow_liteflownetv3.yaml -o data/gtea
```

## Extract Feature(Optional)
```bash
python tools/extract/extract_features.py -c config/extract_feature/extract_feature_i3d_thumos14.yaml -o data/thumos14
```

## Dataset Normalization
```bash
# count mean and std from video
# gtea
python tools/dataset_transform/transform_segmentation_label.py data/gtea data/gtea/groundTruth data/gtea --mode localization --fps 15
python tools/dataset_transform/prepare_video_recognition_data.py data/gtea/label.json data/gtea/Videos data/gtea --negative_sample_num 100 --only_norm True --fps 15 --dataset_type gtea_rgb
python tools/dataset_transform/prepare_video_recognition_data.py data/gtea/label.json data/gtea/flow data/gtea --negative_sample_num 100 --only_norm True --fps 15 --dataset_type gtea_flow

# egtea
python tools/dataset_transform/prepare_video_recognition_data.py data/egtea/egtea.json data/egtea/Videos data/egtea --negative_sample_num 1000 --only_norm True --fps 24 --dataset_type egtea_rgb

# 50salads
python tools/dataset_transform/transform_segmentation_label.py data/50salads data/50salads/groundTruth data/50salads --mode localization --fps 30
python tools/dataset_transform/prepare_video_recognition_data.py data/50salads/label.json data/50salads/Videos data/50salads --negative_sample_num 1000 --only_norm True --fps 30 --dataset_type 50salads_rgb

# thumos14
python tools/dataset_transform/transform_segmentation_label.py data/thumos14/gt.json data/thumos14/Videos data/thumos14 --fps 30

```

Here releases dataset mean and std config

- gtea:
```txt
# rgb
mean RGB :[140.39158961711036, 108.18022223151027, 45.72351736766547]
std RGB :[33.94421369129452, 35.93603536756186, 31.508484434367805]
# flows
mean RGB :[0.9686297051020777, 0.9706158002294017, 0.972493270804535] * 255
std RGB :[0.039060756165796726, 0.03689212641350189, 0.03209093941013171] * 255
```
- egtea:
```txt
mean RGB ∶[0.47882690412518875, 0.30667687330914223, 0.1764174579795214] * 255
std RGB :[0.26380785444954574, 0.20396220265286277, 0.16305419562005563] * 255
```
- 50salads:
```txt
mean RGB ∶[0.5139909998345553, 0.5117725498677757，0.4798814301515671] * 255
std RGB :[0.23608918491478523, 0.23385714300069754, 0.23755006337414028] * 255
```
- breakfast:
```txt
mean RGB ∶[0.4245283568405083, 0.3904851168609079, 0.33709139617292494] * 255
std RGB :[0.26207845745959846, 0.26008439810422, 0.24623600365905168] * 255
```
- thumos14:
```txt
mean RGB ∶[0.384953972862144, 0.38326867429930167, 0.3525199505706894] * 255
std RGB :[0.258450710004705, 0.2544892750057763, 0.24812118173426492] * 255
```
## Convert Localization Label to Segmentation Label
```bash
# egtea
python tools/dataset_transform/transform_egtea_label.py data/egtea/splits_label data/egtea/verb_idx.txt data/egtea
python tools/dataset_transform/transform_segmentation_label.py data/egtea/egtea.json data/egtea/Videos data/egtea --mode segmentation --fps 24
```

## For EGTEA we mannul split test set
### split1 test
```txt
OP01-R01-PastaSalad.mp4
OP01-R02-TurkeySandwich.mp4
OP01-R03-BaconAndEggs.mp4
OP01-R04-ContinentalBreakfast.mp4
OP01-R05-Cheeseburger.mp4
OP01-R06-GreekSalad.mp4
OP01-R07-Pizza.mp4
OP02-R01-PastaSalad.mp4
OP02-R02-TurkeySandwich.mp4
OP02-R03-BaconAndEggs.mp4
OP02-R04-ContinentalBreakfast.mp4
OP02-R05-Cheeseburger.mp4
OP02-R06-GreekSalad.mp4
OP02-R07-Pizza.mp4
P01-R01-PastaSalad.mp4
P01-R02-TurkeySandwich.mp4
P02-R01-PastaSalad.mp4
P02-R03-BaconAndEggs.mp4
P02-R04-ContinentalBreakfast.mp4
P02-R05-Cheeseburger.mp4
P02-R06-GreekSalad.mp4
P03-R01-PastaSalad.mp4
P04-R01-PastaSalad.mp4
P04-R05-Cheeseburger.mp4
P04-R06-GreekSalad.mp4
P05-R01-PastaSalad.mp4
P05-R02-TurkeySandwich.mp4
P06-R01-PastaSalad.mp4
P06-R02-TurkeySandwich.mp4
P07-R01-PastaSalad.mp4
```

### split2 test
```txt
OP03-R01-PastaSalad.mp4
OP03-R02-TurkeySandwich.mp4
OP03-R03-BaconAndEggs.mp4
OP03-R04-ContinentalBreakfast.mp4
OP03-R05-Cheeseburger.mp4
OP03-R06-GreekSalad.mp4
OP03-R07-Pizza.mp4
OP04-R01-PastaSalad.mp4
OP04-R02-TurkeySandwich.mp4
OP04-R03-BaconAndEggs.mp4
OP04-R04-ContinentalBreakfast.mp4
OP04-R05-Cheeseburger.mp4
OP04-R06-GreekSalad.mp4
OP04-R07-Pizza.mp4
P08-R01-PastaSalad.mp4
P09-R01-PastaSalad.mp4
P09-R02-TurkeySandwich.mp4
P10-R01-PastaSalad.mp4
P10-R02-TurkeySandwich.mp4
P10-R05-Cheeseburger.mp4
P10-R06-GreekSalad.mp4
P11-R01-PastaSalad.mp4
P11-R02-TurkeySandwich.mp4
P12-R01-PastaSalad.mp4
P12-R02-TurkeySandwich.mp4
P13-R01-PastaSalad.mp4
P14-R01-PastaSalad.mp4
P14-R02-TurkeySandwich.mp4
```

### split3 test
```txt
OP05-R03-BaconAndEggs.mp4
OP05-R04-ContinentalBreakfast.mp4
OP05-R07-Pizza.mp4
OP06-R02-TurkeySandwich.mp4
OP06-R03-BaconAndEggs.mp4
OP06-R04-ContinentalBreakfast.mp4
OP06-R05-Cheeseburger.mp4
OP06-R06-GreekSalad.mp4
OP06-R07-Pizza.mp4
P15-R01-PastaSalad.mp4
P16-R03-BaconAndEggs.mp4
P17-R03-BaconAndEggs.mp4
P17-R04-ContinentalBreakfast.mp4
P18-R03-BaconAndEggs.mp4
P18-R04-ContinentalBreakfast.mp4
P19-R03-BaconAndEggs.mp4
P19-R04-ContinentalBreakfast.mp4
P20-R03-BaconAndEggs.mp4
P20-R04-ContinentalBreakfast.mp4
P21-R03-BaconAndEggs.mp4
P21-R04-ContinentalBreakfast.mp4
P21-R05-Cheeseburger.mp4
P21-R06-GreekSalad.mp4
P22-R03-BaconAndEggs.mp4
P23-R03-BaconAndEggs.mp4
P24-R03-BaconAndEggs.mp4
P25-R06-GreekSalad.mp4
P26-R05-Cheeseburger.mp4
```