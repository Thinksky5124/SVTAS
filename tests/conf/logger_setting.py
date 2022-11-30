'''
Author       : Thyssen Wen
Date         : 2022-11-30 09:50:09
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-30 09:58:27
Description  : logger conf ref:https://blog.csdn.net/qq_41497271/article/details/114967847
FilePath     : /SVTAS/test/conf/logger_setting.py
'''
import os

standard_format = '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]' \
                  '[%(levelname)s][%(message)s]'

simple_format = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s'

test_format = '%(asctime)s] %(message)s'

# 3、日志配置字典
LOGGING_DIC = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': standard_format
        },
        'simple': {
            'format': simple_format
        },
        'test': {
            'format': test_format
        },
    },
    'filters': {},
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },

        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            # custom path
            # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            # LOG_PATH = os.path.join(BASE_DIR,'a1.log')
            'filename': os.path.join("./output/test_report",'auto_test.log'),
            'maxBytes': 1024*1024*5,
            'backupCount': 5,
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        #logging.getLogger(__name__)
        'test': {
            'handlers': ['default', 'console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}