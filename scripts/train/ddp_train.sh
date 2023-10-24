###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors: Thinksky5124 Thinksky5124@outlook.com
 # @LastEditTime: 2023-05-04 10:48:07
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/ddp_train.sh
### 
export CUDA_VISIBLE_DEVICES=0,1

# mstcn 1538574472
# asformer 19980125
export DECORD_EOF_RETRY_MAX=20480
### gtea ###
torchrun --standalone --nnodes=1 --nproc_per_node=2 tools/launch.py --mode train -c config/svtas/diffact/breakfast/stream_diffact_breakfast_torch_ddp.py --seed 19990924
# torchrun --standalone --nnodes=1 --nproc_per_node=2 tools/launch.py --mode train -c config/svtas/diffact/diffact_gtea_deepspeed.py --seed 0
