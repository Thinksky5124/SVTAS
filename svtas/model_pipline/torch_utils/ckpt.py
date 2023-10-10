'''
Author       : Thyssen Wen
Date         : 2023-10-10 20:47:54
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-10 21:52:08
Description  : ref: https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/checkpoint.py
FilePath     : \ETESVS\svtas\model_pipline\torch_utils\ckpt.py
'''
import re
import logging
from collections import OrderedDict, namedtuple
from svtas.utils.logger import print_log
from svtas.dist import get_dist_info
from typing import Callable, Dict

class _IncompatibleKeys(
        namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__

class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: Dict[str, Callable] = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger='current'):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to None
            logger (str): The logger for message. Defaults to 'current'.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print_log(
            f'Loads checkpoint by {class_name[10:]} backend from path: '
            f'{filename}',
            logger=logger)
        return checkpoint_loader(filename, map_location)
    
def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    missing_keys = []
    err_msg = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, local_state_dict, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))

        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(local_state_dict, prefix, local_metadata,
                                     True, missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, '_load_state_dict_post_hooks'):
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    'Hooks registered with '
                    '``register_load_state_dict_post_hook`` are not expected '
                    'to return new values, if incompatible_keys need to be '
                    'modified, it should be done inplace.')

    load(module, state_dict)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print_log(err_msg, logger=logger, level=logging.WARNING)

def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Defaults to None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
        OrderedDict storing model weights or a dict containing other
        information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)

def _load_checkpoint_to_model(model,
                              checkpoint,
                              strict=False,
                              logger=None,
                              revise_keys=[(r'^module\.', '')]):

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    return _load_checkpoint_to_model(model, checkpoint, strict, logger,
                                     revise_keys)

def _load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict
