###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-09 21:36:58
 # @Description  : Test script
 # @FilePath     : /SVTAS/scripts/test.sh
### 

export CUDA_VISIBLE_DEVICES=0

python tools/launch.py  --mode test -c config/svtas/diffact/gtea/stream_diffact_gtea.py
