
###
 # @Author       : Thyssen Wen
 # @Date         : 2022-10-24 16:10:45
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-10-25 17:21:55
 # @Description  : feature extract script
 # @FilePath     : /SVTAS/script/feature_extract.sh
### 

export CUDA_VISIBLE_DEVICES=1

### gtea ###
python tools/extract/extract_features.py -c config/extract_feature/timesformer_gtea.py -o data/gtea