'''
Author       : Thyssen Wen
Date         : 2022-11-30 10:00:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-01 10:49:37
Description  : Launch Pytest
FilePath     : /SVTAS/tools/launch_pytest.py
'''
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import argparse
from tests.conf import logger_setting
from logging import config, getLogger
config.dictConfig(logger_setting.LOGGING_DIC)
logger = getLogger('test')

def parse_args():
    parser = argparse.ArgumentParser("SVTAS unit test case script")
    parser.add_argument('-p',
                        '--path_report',
                        type=str,
                        default='./output/test_report',
                        help='test report file path')
    parser.add_argument(
        '-v',
        action='store_true',
        help='whether to detail param during testing case')
    parser.add_argument(
        '-s',
        action='store_true',
        help='whether to print execute information during testing case')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='whether to debug testing case')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    add_args = ''
    if not args.debug:
        if args.v:
            add_args = add_args + '-v '
        if args.s:
            add_args = add_args + '-s '
        os.system(f'pytest tests {add_args} --html={args.path_report}/report.html')
        logger.info("All Test Case Finish!")
    else:
        from tests.test_cases.test_sbp import TestSBP
        test_class = TestSBP()
        test_class.test_save_load_model()