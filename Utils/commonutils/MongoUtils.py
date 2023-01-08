# pip install pymongo
# reference https://api.mongodb.com/python/current/tutorial.html http://developer.51cto.com/art/201805/573924.htm
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from urllib.parse import quote_plus
from Utils.commonutils.MysqlUtils import *

def getMongoClient(user: str = 'root', password: str = 'root27017', host: str = '192.168.3.3:27017') -> MongoClient:
    uri = "mongodb://%s:%s@%s" % (
        quote_plus(user), quote_plus(password), host)
    mongo_client: MongoClient = MongoClient(uri)
    return mongo_client


def getDatabase(database: str, mongo_client: MongoClient) -> Database:
    return mongo_client[database]


def getCollection(m_collection: str, database: str, mongo_client: MongoClient) -> Collection:
    db = getDatabase(database, mongo_client)
    return db[m_collection]


def close(mongo_client: MongoClient):
    mongo_client.close()


if __name__ == '__main__':
    # client = getMongoClient()
    # yekTestDB = client.yekTest
    # collection: Collection = yekTestDB['test-collection']  # or yekTestDB.test-collection
    # doc = {'name': 'yek', 'age': 12}
    # id1 = collection.insert_one(doc).inserted_id
    # print(id1)
    # # close(client)
    # s1 = dba.data_inquiry_all('show status like %s', params=('Threads%',))
    # print(s1)
    # result = dba.data_inquiry_all('select * from queue')
    # print(result)
    # s1 = dba.data_inquiry_all('show status like %s', params=('Threads%',))
    # print(s1)
    # import time
    #
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
