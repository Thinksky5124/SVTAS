# Demo
This is a real-time temporal action segmentation run demo.

# Install
- python >= 3.8

## Step 1
```bash
cd demo
pip install -r requirements.txt
```
## Step 2
```bash
python infer.py -m MobileNetV2TSM_3D_TCN_gtea_split1.onnx -i S1_Cheese_C1.mp4 -l mapping.txt -o ./output.mp4 --visualize
```