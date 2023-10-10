'''
Author       : Thyssen Wen
Date         : 2023-10-09 23:24:55
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-10 21:33:49
Description  : ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/import_utils.py
FilePath     : \ETESVS\svtas\utils\import_utils.py
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
- dist need:
    deepspeed
- visulize need:
    grad-cam
    h5py
    tensorboard
- infer need:
    onnx
    onnxruntime
- dev need:
    pytest
"""
import sys
import importlib.util

from ..logger import get_logger
logger = get_logger("SVTAS")



# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

_torch_available = importlib.util.find_spec("torch") is not None
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
        logger.info(f"PyTorch version {_torch_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TORCH is set")
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
    if _onnx_available:
        logger.debug(f"Successfully imported onnxruntime version {_onnxruntime_version}")

_tensorboard_available = importlib.util.find_spec("tensorboard")
try:
    _tensorboard_version = importlib_metadata.version("tensorboard")
    logger.debug(f"Successfully imported tensorboard version {_tensorboard_version}")
except importlib_metadata.PackageNotFoundError:
    _tensorboard_available = False

_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
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
    if _opencv_available:
        logger.debug(f"Successfully imported cv2 version {_opencv_version}")
except importlib_metadata.PackageNotFoundError:
    _opencv_available = False

_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    _scipy_version = importlib_metadata.version("scipy")
    logger.debug(f"Successfully imported scipy version {_scipy_version}")
except importlib_metadata.PackageNotFoundError:
    _scipy_available = False

_tqdm_available = importlib.util.find_spec("tqdm") is not None
try:
    _tqdm_version = importlib_metadata.version("tqdm")
    logger.debug(f"Successfully imported tqdm version {_tqdm_version}")
except importlib_metadata.PackageNotFoundError:
    _tqdm_available = False

_torchvision_available = importlib.util.find_spec("torchvision") is not None
try:
    _torchvision_version = importlib_metadata.version("torchvision")
    logger.debug(f"Successfully imported torchvision version {_torchvision_version}")
except importlib_metadata.PackageNotFoundError:
    _torchvision_available = False

_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
try:
    _torchaudio_version = importlib_metadata.version("torchaudio")
    logger.debug(f"Successfully imported torchaudio version {_torchaudio_version}")
except importlib_metadata.PackageNotFoundError:
    _torchaudio_available = False

_num2words_available = importlib.util.find_spec("num2words") is not None
try:
    _num2words_version = importlib_metadata.version("num2words")
    logger.debug(f"Successfully imported num2words version {_num2words_version}")
except importlib_metadata.PackageNotFoundError:
    _num2words_available = False

_einops_available = importlib.util.find_spec("einops") is not None
try:
    _einops_version = importlib_metadata.version("einops")
    logger.debug(f"Successfully imported einops version {_einops_version}")
except importlib_metadata.PackageNotFoundError:
    _einops_available = False

_sklearn_available = importlib.util.find_spec("sklearn") is not None
try:
    _sklearn_version = importlib_metadata.version("sklearn")
    logger.debug(f"Successfully imported sklearn version {_sklearn_version}")
except importlib_metadata.PackageNotFoundError:
    _sklearn_available = False

_pillow_available = importlib.util.find_spec("pillow") is not None
try:
    _pillow_version = importlib_metadata.version("pillow")
    logger.debug(f"Successfully imported pillow version {_pillow_version}")
except importlib_metadata.PackageNotFoundError:
    _pillow_available = False

_ffmpy_available = importlib.util.find_spec("ffmpy") is not None
try:
    _ffmpy_version = importlib_metadata.version("ffmpy")
    logger.debug(f"Successfully imported ffmpy version {_ffmpy_version}")
except importlib_metadata.PackageNotFoundError:
    _ffmpy_available = False

_deepspeed_available = importlib.util.find_spec("deepspeed") is not None
try:
    _deepspeed_version = importlib_metadata.version("deepspeed")
    logger.debug(f"Successfully imported deepspeed version {_deepspeed_version}")
except importlib_metadata.PackageNotFoundError:
    _deepspeed_available = False

_pytorch_grad_cam_available = importlib.util.find_spec("pytorch_grad_cam") is not None
try:
    _pytorch_grad_cam_version = importlib_metadata.version("pytorch_grad_cam")
    logger.debug(f"Successfully imported pytorch_grad_cam version {_pytorch_grad_cam_version}")
except importlib_metadata.PackageNotFoundError:
    _pytorch_grad_cam_available = False

_h5py_available = importlib.util.find_spec("h5py") is not None
try:
    _h5py_version = importlib_metadata.version("h5py")
    logger.debug(f"Successfully imported h5py version {_h5py_version}")
except importlib_metadata.PackageNotFoundError:
    _h5py_available = False

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