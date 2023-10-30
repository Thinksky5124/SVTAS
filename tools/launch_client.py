import argparse
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)

from svtas.utils import AbstractBuildFactory, get_logger, mkdir
from svtas.utils.config import get_config

def parse_args():
    parser = argparse.ArgumentParser("SVTAS serving script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.py',
                        help='config file path')
    parser.add_argument('-o',
                    '--override',
                    action='append',
                    default=[],
                    help='config options to be overridden')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = "-1"
        os.environ['WORLD_SIZE'] = "1"
    return args

def main():
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # build client
    client = AbstractBuildFactory.create_factory('serving_client').create(cfg.CLIENT)
    
    # run
    client.init_client()
    client.run()
    client.shutdown()

if __name__ == '__main__':
    main()