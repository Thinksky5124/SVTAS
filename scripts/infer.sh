###
 # @Author       : Thyssen Wen
 # @Date         : 2022-09-24 20:54:46
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-22 16:54:45
 # @Description  : Infer script
 # @FilePath     : /SVTAS/scripts/infer.sh
### 

# infer
# python tools/launch.py --mode infer -c config/infer/swinv2_infer_tensorrt.py
python tools/launch.py --mode infer -c config/infer/swinv2_infer_torch.py