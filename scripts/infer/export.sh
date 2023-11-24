###
 # @Author       : Thyssen Wen
 # @Date         : 2022-09-24 20:54:46
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-31 10:28:39
 # @Description  : Infer script
 # @FilePath     : /SVTAS/scripts/infer/export.sh
### 

# export model tensorRT
# python tools/launch.py --mode export -c config/export/swinv2_export_tensorRT.py
python tools/launch.py --mode export -c config/export/test_onnx.py
# export model TVM
# python tools/launch.py --mode export -c config/export/swinv2_export_tvm.py
# export model onnx
# python tools/launch.py --mode export -c config/export/swinv2_export_onnx.py