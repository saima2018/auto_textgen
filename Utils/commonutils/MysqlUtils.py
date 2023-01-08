# -*- coding: utf8 -*-
"""
数据库连接池管理模块
"""
import pymysql
from DBUtils.PooledDB import PooledDB
from conf import DB_config as Config
from functools import wraps
# 使用连接池之前的连接方式
# def connectMysql(host='192.168.3.3', port=13306, user='root', password='bat100', database='companyKG'):
#     """
#     链接到mysql 返回连接对象和光标
#     :return:
#     """
#     # 连接database
#     conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database)
#     # 得到一个可以执行sql语句的光标对象
#     cursor = conn.cursor()
#     return conn, cursor
#
#
# def connClose(conn, cursor):
#     """
#     关闭连接，释放资源
#     :param conn:
#     :param cursor:
#     :return:
#     """
#     cursor.close()
#     conn.close

class PTConnectionPool(object):
    __pool = None
    def __enter__(self):
        self.conn = self.__getConn()
        self.cursor = self.conn.cursor()
        return self

    def __getConn(self):
        if self.__pool is None:
            self.__pool = PooledDB(creator=pymysql, mincached=Config.DB_MIN_CACHED,maxcached=Config.DB_MAX_CACHED,
                                   maxshared=Config.DB_MAX_SHARED, maxconnections=Config.DB_MAX_CONNECYIONS,
                                   blocking=Config.DB_BLOCKING, maxusage=Config.DB_MAX_USAGE,
                                   setsession=Config.DB_SET_SESSION,
                                   host=Config.DB_HOST , port=Config.DB_PORT ,
                                   user=Config.DB_USER , passwd=Config.DB_PASSWORD ,
                                   db=Config.DB_DBNAME, use_unicode=False, charset="utf8")

        return self.__pool.connection()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        self.conn.close()
def getPTConnection():
    return PTConnectionPool()

# 单例测试
# def singleton(cls):
#     instances={}
#     @wraps(cls)
#     def get_instance(*args, **kw):
#         if cls not in instances:
#             instances[cls] = cls(*args, **kw)
#         return instances[cls]
#     return get_instance()
#
# db_pool_ins = None
# @singleton
# class DBPool():
#     def __init__(self):
#         self.pool = PooledDB(creator=pymysql, mincached=Config.DB_MIN_CACHED, maxcached=Config.DB_MAX_CACHED,
#                                maxshared=Config.DB_MAX_SHARED, maxconnections=Config.DB_MAX_CONNECYIONS,
#                                blocking=Config.DB_BLOCKING, maxusage=Config.DB_MAX_USAGE,
#                                setsession=Config.DB_SET_SESSION,
#                                host=Config.DB_HOST, port=Config.DB_PORT,
#                                user=Config.DB_USER, passwd=Config.DB_PASSWORD,
#                                db=Config.DB_DBNAME, use_unicode=False, charset="utf8")
#
#     def get_connection(self):
#         return self.pool.connection()
# class DBAction():
#     def __init__(self):
#         global db_pool_ins
#         if db_pool_ins == None:
#             db_pool_ins = DBPool()
#         self.conn = db_pool_ins.get_connection()
#         self.cursor = self.conn.cursor()
#     def close_database(self):
#         self.cursor.close()
#         self.conn.close()
#     def data_inquiry_all(self,sql,params=()):
#         self.cursor.execute(sql,params)
#         result = self.cursor.fetchall()
#         return result

dba=getPTConnection()

if __name__ == "__main__":
    # s1 = dba.data_inquiry_all('show status like %s',params=('Threads%',))
    # print(s1)
    # result = dba.data_inquiry_all('select * from queue')
    # print(result)
    # s1 = dba.data_inquiry_all('show status like %s',params=('Threads%',))
    # print(s1)
    # import time
    # time.sleep(5)
    # s1 = dba.data_inquiry_all('show status like %s', params=('Threads%',))
    # print(s1)
    with dba as db:
        sql_status = "show status like 'Threads%'"

        db.cursor.execute(sql_status)
        a = db.cursor.fetchall()
        print(a)
        import time

        time.sleep(5)
        sql_status = "show status like 'Threads%'"

        db.cursor.execute(sql_status)
        a = db.cursor.fetchall()
        print(a)

