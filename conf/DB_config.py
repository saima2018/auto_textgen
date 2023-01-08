# -*- coding: utf-8 -*-
"""
MYSQL数据库连接配置
"""
import sys
import os
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)
from Utils.DataUtils import load_xml_conf

# 从配置文件中获取数据库参数
conf = load_xml_conf()
db = conf['MysqlUtils_Database']['db']


#数据库信息
DB_HOST=db['host']
DB_PORT=db['port']
DB_DBNAME=db['database']
DB_USER=db['user']
DB_PASSWORD=db['password']


#数据库连接编码
DB_CHARSET="utf8"

#mincached : 启动时开启的闲置连接数量(缺省值 0 以为着开始时不创建连接)
DB_MIN_CACHED=10

#maxcached : 连接池中允许的闲置的最多连接数量(缺省值 0 代表不闲置连接池大小)
DB_MAX_CACHED=10

#maxshared : 共享连接数允许的最大数量(缺省值 0 代表所有连接都是专用的)如果达到了最大数量,被请求为共享的连接将会被共享使用
DB_MAX_SHARED=20

#maxconnecyions : 创建连接池的最大数量(缺省值 0 代表不限制)
DB_MAX_CONNECYIONS=100

#blocking : 设置在连接池达到最大数量时的行为(缺省值 0 或 False 代表返回一个错误<toMany......> 其他代表阻塞直到连接数减少,连接被分配)
DB_BLOCKING=True

#maxusage : 单个连接的最大允许复用次数(缺省值 0 或 False 代表不限制的复用).当达到最大数时,连接会自动重新连接(关闭和重新打开)
DB_MAX_USAGE=0

#setsession : 一个可选的SQL命令列表用于准备每个会话，如["set datestyle to german", ...]
DB_SET_SESSION=None