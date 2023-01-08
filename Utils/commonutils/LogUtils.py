# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@version:
@author: fzq
@contact: fanzhiqiang@bat100.net
@software: PyCharm
@file: LogUtils.py
@time: 2020/3/13 17:09
@description: 日志类
"""

import os
import traceback
import sys
from logging import config, getLogger


class LogUtils(object):
    # 是否初始化
    is_init = False

    def __init__(self):
        if not LogUtils.is_init:
            conf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "configure/logging.conf")
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            config.fileConfig(conf_path, defaults={'logfilename': log_path})
            LogUtils.is_init = True

    def __new__(cls, *args, **kwargs):
        """
        单例模式
        """
        if not hasattr(cls, 'instance'):
            cls.instance = super(LogUtils, cls).__new__(cls)
        return cls.instance

    def getLogger(self, log_name='root'):
        logger = getLogger(log_name)
        return logger

    def getLogException(self, logger, loginfo, info_type='error', error_info=''):
        """
        异常处理，并将异常结果的 异常类型、异常内容、异常信息等写入log日志文件
        :param loginfo: loginfo: sys.exc_info()
        :param info_type:
        :return: 无返回，将信息计入日志
        """
        log_info = 'ErrorType:{0}, ErrorContent:{1},  ErrorInfo:{2}'
        exc_type, exc_value, exc_traceback_obj = loginfo
        abnormal_info = ''.join(traceback.format_tb(exc_traceback_obj))
        if info_type == 'error':
            logger.error(log_info.format(exc_type.__name__, exc_value, abnormal_info))
        elif info_type == 'input_error':
            logger.error(log_info.format('input_error', '', error_info))
        elif info_type == 'warn':
            logger.warn(log_info.format(exc_type.__name__, exc_value, abnormal_info))


if __name__ == '__main__':
    app = LogUtils()
    logger = app.getLogger()
    a = {'a': 1, 'b': 2}
    try:
        print(a['c'])
    except Exception:
        app.getLogException(logger, sys.exc_info(), info_type='error')
