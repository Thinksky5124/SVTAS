# setup
The correlation layer is borrowed from [NVIDIA-flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)

if report `RuntimeError: CUDA error: the provided PTX was compiled with an unsupported toolchain.`, you should modify file `setup.py`'s `nvcc_args` for your gpus arch. More detail from [issuse](https://github.com/rusty1s/pytorch_scatter/issues/225)


```bash
pip install cupy
cd correlation_package
python setup.py install
```