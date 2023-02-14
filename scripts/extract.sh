
###
 # @Author       : Thyssen Wen
 # @Date         : 2022-10-24 16:10:45
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-02-13 14:36:13
 # @Description  : feature extract script
 # @FilePath     : /SVTAS/scripts/extract.sh
### 

export CUDA_VISIBLE_DEVICES=0

### gtea rgb feature ###
python tools/extract/extract_features.py -c config/extract/extract_feature/swin_transformer_3d_sbp_gtea.py -o data/gtea
# python tools/extract/extract_features.py -c config/extract/extract_feature/mobilev2_tsm_rgb_gtea.py -o data/gtea
### gtea flow ###
# python tools/extract/extract_flow.py -c config/extract/extract_flow/raft_gtea.py -o data/gtea
### gtea flow feature ###
# python tools/extract/extract_features.py -c config/extract/extract_feature/i3d_r50_flow_gtea.py -o data/gtea --flow_extract
### gtea mvs res ###
# python tools/extract/extract_mvs_res.py -c config/extract/extract_mvs_res/mvs_res_gtea.py -o data/gtea