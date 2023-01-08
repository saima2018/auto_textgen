# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@version:
@author: fzq
@contact: fanzhiqiang@bat100.net
@software: PyCharm
@file: TransUtils.py
@time: 2019/8/7 15:06
@description:本模块的作用是提供一些数据格式处理、转换的工具
"""


def full2Half(ustring):
    """
    全角转半角
    :param ustring:
    :return:
    """
    ss = []
    for s in ustring:
        inside_code = ord(s)
        if inside_code == 12288:
            # 空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            # 其它按公式进行转换
            inside_code -= 65248
        ss.append(chr(inside_code))
    return ''.join(ss)


def half2Full(ustring):
    """
    半角转全角
    :param ustring:
    :return:
    """
    ss = []
    for s in ustring:
        inside_code = ord(s)
        if inside_code == 32:
            # 空格直接转换
            inside_code = 12288
        elif 33 <= inside_code <= 126:
            # 其它按公式进行转换
            inside_code += 65248
        ss.append(chr(inside_code))
    return ''.join(ss)


def chi2Eng(tmp_str):
    """
    中文标点符号转换成英文标点符号
    :param tmp_str:
    :return:
    """
    table = {ord(f): ord(t) for f, t in zip('，（）：；', ',():;')}
    return tmp_str.translate(table)


def eng2Chi(tmp_str):
    """
    英文标点符号转换为中文标点符号
    :param tmp_str:
    :return:
    """
    table = {ord(f): ord(t) for f, t in zip(',():;', '，（）：；')}
    return tmp_str.translate(table)


if __name__ == '__main__':
    a = '生产ＰＶＣ系列产品}，产品（９０％）外销。'
    # print('全角：', half2Full(a))
    # print('半角：', full2Half(a))
    # b = full2Half(a)
    # import re
    # print(re.sub('（', '@@', a))
    # print(re.sub('（', '@@', b))
    print(chi2Eng(a))