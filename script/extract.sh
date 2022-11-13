
###
 # @Author       : Thyssen Wen
 # @Date         : 2022-10-24 16:10:45
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-13 09:27:56
 # @Description  : feature extract script
 # @FilePath     : /SVTAS/script/extract.sh
### 

export CUDA_VISIBLE_DEVICES=0

### gtea rgb feature ###
# python tools/extract/extract_features.py -c config/extract/extract_feature/i3d_rgb_gtea.py -o data/gtea
### gtea flow ###
# python tools/extract/extract_flow.py -c config/extract/extract_flow/raft_gtea.py -o data/gtea
### gtea flow feature ###
# python tools/extract/extract_features.py -c config/extract/extract_feature/i3d_flow_gtea.py -o data/gtea --flow_extract
### gtea mvs res ###
python tools/extract/extract_mvs_res.py -c config/extract/extract_mvs_res/mvs_res_gtea.py -o data/gtea