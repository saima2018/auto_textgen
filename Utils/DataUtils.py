# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@version:
@author: fzq
@contact: fanzhiqiang@bat100.net
@software: PyCharm
@file: DataUtils.py
@time: 2020/3/4 13:59
@description:
"""

import os
import paramiko
# from encoder import XML2Dict
import pymysql

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_xml_conf():
    """
    Parse xml format configuration file
    :return: configuration in json format
    """
    configure_file = os.path.join(root_path, 'conf', 'config.xml')
    info = open(configure_file, encoding='utf-8').read()
    info = XML2Dict().parse(info)['configuration']
    res = {}
    for info_i in info['property']:
        key_name = info_i['name'].decode('utf-8')
        res[key_name] = eval(info_i['value'].decode('utf-8'))
    return res



def remote_scp(host, remote_path, local_path, username, password):
    """
    从远端服务器上下载文件
    :param host: 远端服务器ip地址
    :param remote_path: 远端服务器上文件全路径
    :param local_path: 本地文件全路径
    :param username: 远端服务器登录用户名
    :param password: 远端服务器登录密码
    :return:
    """
    # 默认使用22端口
    t = paramiko.Transport(host, 22)
    # 登录远程服务器
    t.connect(username=username, password=password)
    # sftp传输协议
    sftp = paramiko.SFTPClient.from_transport(t)
    src = remote_path
    des = local_path
    sftp.get(src, des)
    t.close()

class XML2Dict(object):

    def __init__(self, coding='UTF-8'):
        self._coding = coding

    def _parse_node(self, node):
        tree = {}

        #Save childrens
        for child in node.getchildren():
            ctag = child.tag
            cattr = child.attrib
            ctext = child.text.strip().encode(self._coding) if child.text is not None else ''
            ctree = self._parse_node(child)

            if not ctree:
                cdict = self._make_dict(ctag, ctext, cattr)
            else:
                cdict = self._make_dict(ctag, ctree, cattr)

            if ctag not in tree: # First time found
                tree.update(cdict)
                continue

            atag = '@' + ctag
            atree = tree[ctag]
            if not isinstance(atree, list):
                if not isinstance(atree, dict):
                    atree = {}

                if atag in tree:
                    atree['#'+ctag] = tree[atag]
                    del tree[atag]

                tree[ctag] = [atree] # Multi entries, change to list

            if cattr:
                ctree['#'+ctag] = cattr

            tree[ctag].append(ctree)

        return  tree

if __name__ == '__main__':
    conf = load_xml_conf()
    print(conf['task_scheduler'])

    # remote_scp(host='192.168.3.80',
    #            remote_path='/home/ubuntu/server.log',
    #            local_path=root_path + 'data/下载.txt',
    #            username='ubuntu',
    #            password='1')
