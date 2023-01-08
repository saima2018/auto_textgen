# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@version:
@author: fzq
@contact: fanzhiqiang@bat100.net
@software: PyCharm
@file: IoUtils.py
@time: 2019/8/7 15:06
@description:本模块的作用是提供IO相关的工具
"""

import json
import os
import csv


def saveJson(out_json, out_file, out_format=False):
    """
    将json数据写入到文件
    :param out_json: 待输出的json格式数据
    :param out_file: 输出文件地址
    :param out_format: 是否进行格式化输出
    :return:
    """
    existFileDir(out_file)
    if out_format:
        json.dump(out_json, open(out_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    else:
        json.dump(out_json, open(out_file, 'w', encoding='utf-8'), ensure_ascii=False)


def loadJson(in_file):
    """
    从文件中读取json文件
    :param in_file:
    :return:
    """
    if os.path.exists(in_file):
        with open(in_file, 'r', encoding='utf-8') as g:
            content = g.read()
            if content.startswith(u'\ufeff'):
                content = content.encode('utf8')[3:].decode('utf8')

        return json.loads(content)
    else:
        print('文件不存在，返回空json串')
        return {}


def loadList(in_file):
    """
    从文件中读取list文件
    :param in_file:
    :return:
    """
    if os.path.exists(in_file):
        return json.load(open(in_file, 'r', encoding='utf-8'))
    else:
        print('文件不存在，返回空list')
        return []


def existFileDir(out_file):
    """
    判断要输出的文件目录是否存在，如果不存在则新建
    :param out_file:
    :return:
    """
    file_path, temp_file = os.path.split(out_file)
    if not os.path.exists(file_path):
        print('==== Warn ==== 目录："%s" 不存在,将新建此目录。' % file_path)
        os.makedirs(file_path)


def getFileAndDirNames(in_dir):
    """
    获取某一目录下的所有目录名和字段名
    :param in_dir:
    :return:
    """
    list_dir = []
    list_file = []
    for tmp in os.listdir(in_dir):
        full_path = os.path.join(in_dir, tmp)
        if os.path.isfile(full_path):
            list_file.append(tmp)
        else:
            list_dir.append(tmp)
    return list_file, list_dir


def writeCsv(out_list, out_csvfile):
    """
    将一个列表写入到csv文件中
    :param out_list: 
    :param out_csvfile: 
    :return: 
    """""
    existFileDir(out_csvfile)
    out = open(out_csvfile, 'w', newline='', encoding='utf-8')
    writer = csv.writer(out)
    for tmp_list in out_list:
        writer.writerow(tmp_list)

