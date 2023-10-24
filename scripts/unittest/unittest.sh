###
 # @Author       : Thyssen Wen
 # @Date         : 2022-11-30 09:37:35
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-30 13:40:03
 # @Description  : Unit Test Script
 # @FilePath     : /SVTAS/test/scripts/test.sh
### 
export CUDA_VISIBLE_DEVICES=1
python tools/launch_pytest.py -v -s