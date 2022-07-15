###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-07-15 15:06:03
 # @Description  : train script
 # @FilePath     : /ETESVS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###

# python main.py --validate -c config/gtea/transeger/transeger_split1.yaml --seed 0
# python main.py --validate -c config/gtea/transeger/transeger_split2.yaml --seed 0
# python main.py --validate -c config/gtea/transeger/transeger_split3.yaml --seed 0
# python main.py --validate -c config/gtea/transeger/transeger_split4.yaml --seed 0

# python main.py --validate -c config/gtea/I3D_mstcn/i3d_mstcn_split1.yaml --seed 0
# python main.py --validate -c config/gtea/I3D_mstcn/i3d_mstcn_split2.yaml --seed 0
# python main.py --validate -c config/gtea/I3D_mstcn/i3d_mstcn_split3.yaml --seed 0
# python main.py --validate -c config/gtea/I3D_mstcn/i3d_mstcn_split4.yaml --seed 0

# python main.py --validate -c config/gtea/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split1.yaml --seed 0
# python main.py --validate -c config/gtea/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split2.yaml --seed 0
# python main.py --validate -c config/gtea/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split3.yaml --seed 0
# python main.py --validate -c config/gtea/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split4.yaml --seed 0

### 50salads ###
# python main.py --validate -c config/50salads/transeger/transeger_split1.yaml --seed 0
# python main.py --validate -c config/50salads/transeger/transeger_split2.yaml --seed 0
# python main.py --validate -c config/50salads/transeger/transeger_split3.yaml --seed 0
# python main.py --validate -c config/50salads/transeger/transeger_split4.yaml --seed 0
# python main.py --validate -c config/50salads/transeger/transeger_split5.yaml --seed 0

# python main.py --validate -c config/50salads/I3D_mstcn/i3d_mstcn_split1.yaml --seed 0
# python main.py --validate -c config/50salads/I3D_mstcn/i3d_mstcn_split2.yaml --seed 0
# python main.py --validate -c config/50salads/I3D_mstcn/i3d_mstcn_split3.yaml --seed 0
python main.py --validate -c config/50salads/I3D_mstcn/i3d_mstcn_split4.yaml --seed 0
python main.py --validate -c config/50salads/I3D_mstcn/i3d_mstcn_split5.yaml --seed 0

python main.py --validate -c config/50salads/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split1.yaml --seed 0
python main.py --validate -c config/50salads/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split2.yaml --seed 0
python main.py --validate -c config/50salads/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split3.yaml --seed 0
python main.py --validate -c config/50salads/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split4.yaml --seed 0
python main.py --validate -c config/50salads/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split5.yaml --seed 0

### breakfast ###