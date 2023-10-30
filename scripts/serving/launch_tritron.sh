###
 # @Author       : Thyssen Wen
 # @Date         : 2023-10-22 20:02:49
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-30 10:55:17
 # @Description  : file content
 # @FilePath     : /SVTAS/scripts/serving/launch_tritron.sh
### 
export CUDA_VISIBLE_DEVICES=0

# run
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/wenwujun/SVTAS/output/model_repository:/models nvcr.io/nvidia/tritonserver:23.09-py3 tritonserver --model-repository=/models

# in
# docker run --gpus=1 -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/wenwujun/SVTAS/output/model_repository:/models nvcr.io/nvidia/tritonserver:23.09-py3 tritonserver --model-repository=/models