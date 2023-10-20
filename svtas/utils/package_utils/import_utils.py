'''
Author       : Thyssen Wen
Date         : 2023-10-09 23:24:55
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 21:00:21
Description  : ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/import_utils.py
FilePath     : /SVTAS/svtas/utils/package_utils/import_utils.py
'''
"""
- core need:
    torch
    opencv
    tqdm
    regex
    ftfy
    torchvision
    torchaudio
    num2words
    einops
    sklearn
    Pillow
    ffmpy
    addict
    numpy
    pyyaml
    regex;sys_platform=='win32'
    rich
    termcolor
    yapf
- dist need:
    deepspeed
- visulize need:
    grad-cam
    h5py
    tensorboard
    matplotlib
- infer need:
    onnx
    onnxruntime
    tensorrt
- dev need:
    pytest
"""
import sys
import importlib.util

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

_torch_available = importlib.util.find_spec("torch") is not None
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False
else:
    _torch_available = False

_onnxruntime_version = "N/A"
_onnx_available = importlib.util.find_spec("onnxruntime") is not None
if _onnx_available:
    candidates = (
        "onnxruntime",
        "onnxruntime-gpu",
        "ort_nightly_gpu",
        "onnxruntime-directml",
        "onnxruntime-openvino",
        "ort_nightly_directml",
        "onnxruntime-rocm",
        "onnxruntime-training",
    )
    _onnxruntime_version = None
    # For the metadata, we have to look for both onnxruntime and onnxruntime-gpu
    for pkg in candidates:
        try:
            _onnxruntime_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _onnx_available = _onnxruntime_version is not None

_tensorboard_available = importlib.util.find_spec("tensorboard")
try:
    _tensorboard_version = importlib_metadata.version("tensorboard")
except importlib_metadata.PackageNotFoundError:
    _tensorboard_available = False

_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
except importlib_metadata.PackageNotFoundError:
    _ftfy_available = False

# (sayakpaul): importlib.util.find_spec("opencv-python") returns None even when it's installed.
# _opencv_available = importlib.util.find_spec("opencv-python") is not None
try:
    candidates = (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    _opencv_version = None
    for pkg in candidates:
        try:
            _opencv_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _opencv_available = _opencv_version is not None
except importlib_metadata.PackageNotFoundError:
    _opencv_available = False

_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    _scipy_version = importlib_metadata.version("scipy")
except importlib_metadata.PackageNotFoundError:
    _scipy_available = False

_tqdm_available = importlib.util.find_spec("tqdm") is not None
try:
    _tqdm_version = importlib_metadata.version("tqdm")
except importlib_metadata.PackageNotFoundError:
    _tqdm_available = False

_torchvision_available = importlib.util.find_spec("torchvision") is not None
try:
    _torchvision_version = importlib_metadata.version("torchvision")
except importlib_metadata.PackageNotFoundError:
    _torchvision_available = False

_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
try:
    _torchaudio_version = importlib_metadata.version("torchaudio")
except importlib_metadata.PackageNotFoundError:
    _torchaudio_available = False

_num2words_available = importlib.util.find_spec("num2words") is not None
try:
    _num2words_version = importlib_metadata.version("num2words")
except importlib_metadata.PackageNotFoundError:
    _num2words_available = False

_einops_available = importlib.util.find_spec("einops") is not None
try:
    _einops_version = importlib_metadata.version("einops")
except importlib_metadata.PackageNotFoundError:
    _einops_available = False

_sklearn_available = importlib.util.find_spec("sklearn") is not None
try:
    _sklearn_version = importlib_metadata.version("sklearn")
except importlib_metadata.PackageNotFoundError:
    _sklearn_available = False

_pillow_available = importlib.util.find_spec("pillow") is not None
try:
    _pillow_version = importlib_metadata.version("pillow")
except importlib_metadata.PackageNotFoundError:
    _pillow_available = False

_ffmpy_available = importlib.util.find_spec("ffmpy") is not None
try:
    _ffmpy_version = importlib_metadata.version("ffmpy")
except importlib_metadata.PackageNotFoundError:
    _ffmpy_available = False

_deepspeed_available = importlib.util.find_spec("deepspeed") is not None
try:
    _deepspeed_version = importlib_metadata.version("deepspeed")
except importlib_metadata.PackageNotFoundError:
    _deepspeed_available = False

_pytorch_grad_cam_available = importlib.util.find_spec("pytorch_grad_cam") is not None

_h5py_available = importlib.util.find_spec("h5py") is not None
try:
    _h5py_version = importlib_metadata.version("h5py")
except importlib_metadata.PackageNotFoundError:
    _h5py_available = False

_mmcv_available = importlib.util.find_spec("mmcv") is not None
try:
    _mmcv_version = importlib_metadata.version("mmcv")
except importlib_metadata.PackageNotFoundError:
    _mmcv_available = False

_fvcore_available = importlib.util.find_spec("fvcore") is not None
try:
    _fvcore_version = importlib_metadata.version("fvcore")
except importlib_metadata.PackageNotFoundError:
    _fvcore_available = False

_matplotlib_available = importlib.util.find_spec("matplotlib") is not None
try:
    _matplotlib_version = importlib_metadata.version("matplotlib")
except importlib_metadata.PackageNotFoundError:
    _matplotlib_available = False

_av_available = importlib.util.find_spec("av") is not None
try:
    _av_version = importlib_metadata.version("av")
except importlib_metadata.PackageNotFoundError:
    _av_available = False

_seaborn_available = importlib.util.find_spec("seaborn") is not None
try:
    _seaborn_version = importlib_metadata.version("seaborn")
except importlib_metadata.PackageNotFoundError:
    _seaborn_available = False

_tensorrt_available = importlib.util.find_spec("tensorrt") is not None
try:
    _tensorrt_version = importlib_metadata.version("tensorrt")
except importlib_metadata.PackageNotFoundError:
    _tensorrt_available = False

def is_tensorrt_available():
    return _tensorrt_available

def is_seaborn_available():
    return _seaborn_available
    
def is_av_available():
    return _av_available
    
def is_matplotlib_available():
    return _matplotlib_available

def is_fvcore_available():
    return _fvcore_available

def is_mmcv_available():
    return _mmcv_available

def is_pytorch_grad_cam_available():
    return _pytorch_grad_cam_available

def is_h5py_available():
    return _h5py_available
 
def is_deepspeed_available():
    return _deepspeed_available

def is_ffmpy_available():
    return _ffmpy_available

def is_pillow_available():
    return _pillow_available

def is_sklearn_available():
    return _sklearn_available

def is_einops_available():
    return _einops_available

def is_num2words_available():
    return _num2words_available

def is_torchvision_available():
    return _torchvision_available

def is_torchaudio_available():
    return _torchaudio_available

def is_torch_available():
    return _torch_available

def is_onnx_available():
    return _onnx_available

def is_opencv_available():
    return _opencv_available

def is_scipy_available():
    return _scipy_available

def is_tensorboard_available():
    return _tensorboard_available

def is_ftfy_available():
    return _ftfy_available

def is_tqdm_available():
    return _tqdm_available