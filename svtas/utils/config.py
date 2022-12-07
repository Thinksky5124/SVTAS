'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors: Thyssen Wen
LastEditTime: 2022-04-09 16:05:07
Description: config load function
FilePath: /ETESVS/utils/config.py
'''
import os
from .logger import coloring, get_logger, setup_logger
from mmcv import Config

def get_config(fname, overrides=None, show=True, tensorboard=False, logger_path="output"):
    """
    Read config from file
    """
    assert os.path.exists(fname), ('config file({}) is not exist'.format(fname))
    config = Config.fromfile(fname)
    if "work_dir" not in config:
        config.work_dir = "output"

    if os.path.isabs(config.work_dir):
        os.environ['ROS_LOG_DIR'] = config.work_dir
    else:
        os.environ['ROS_LOG_DIR'] = os.path.join(os.getcwd(), config.work_dir)

    logger = setup_logger(f"./"+ logger_path + f"/{config.model_name}", name="SVTAS", level="INFO", tensorboard=tensorboard)
    override_config(config, overrides)
    if show:
        print_config(config)
    return config

def override(dl, ks, v):
    """
    Recursively replace dict of list
    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """
    logger = get_logger("SVTAS")
    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    assert len(ks) > 0, ('lenght of keys should larger than 0')
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            #assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                logger.warning('A new filed ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            assert ks[0] in dl, (
                '({}) doesn\'t exist in {}, a new dict field is invalid'.format(
                    ks[0], dl))
            override(dl[ks[0]], ks[1:], v)
            
def override_config(config, options=None):
    """
    Recursively override the config
    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                epochs=20',
                'PIPELINE.train.transform.1.ResizeImage.resize_short=300'
            ]
    Returns:
        config(dict): replaced config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt,
                              str), ("option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)

    return config

def print_config(config):
    """
    visualize configs
    Arguments:
        config: configs
    """
    print_dict(config)

def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    logger = get_logger("SVTAS")
    placeholder = "-" * 60
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", coloring(k,
                                                                   "HEADER")))
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ",
                                         coloring(str(k), "HEADER")))
            for value in v:
                print_dict(value, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ",
                                           coloring(k, "HEADER"),
                                           coloring(v, "OKGREEN")))

        if k.isupper():
            logger.info(placeholder)