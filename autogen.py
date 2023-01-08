# -*- coding: UTF-8 -*-
import sys
import os
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)
from flask import Flask
from flask_restful import Api, Resource, reqparse
import json
from generate.bert_gen import bert_gen, domain_model_finetune
from generate.gpt2_gen import gpt2_gen
import traceback
import time
from Utils.commonutils.MysqlUtils import *
from multiprocessing import Process

import logging
import Utils.logger_set as logger_set
# 设定日志文件
logger_set.setup_logging()
logger = logging.getLogger(__file__)

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("input_task",type=str,help='组任务参数，列表套列表，或者单个列表')

import argparse
base_parser = argparse.ArgumentParser('传入参数')
base_parser.add_argument('--host', type=str)
base_parser.add_argument('--port', type=int)
base_parser.add_argument('--gpu',  type=str)
base_args = base_parser.parse_args()
print('==============',type(base_args.gpu),base_args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = base_args.gpu

# 训练模型
def finetune(task_id, model_id, corpus_full_path):
    """model_id是被训练的已有模型id"""
    try:
        # 从model表中读取待训练模型路径和名称
        sql_model = "SELECT model_size, model_file_path, model_file_name, model_name FROM model WHERE id = %s"
        with dba as db:
            try:
                db.cursor.execute(sql_model, model_id)
                params = db.cursor.fetchall()
                params = params[0]
                finetune_model_dir_before = params[1]
                finetune_model_name_before = params[2]
            except:
                logger.error('数据库连接有误')
                logger.error(str(traceback.format_exc()))

        # 自动设定训练后模型名称finetune_model_name_after
        if finetune_model_dir_before == 'models': # models目录存放原始模型，自动按创建时间命名
            finetune_model_name_after = 'finetuned_model' + str(time.strftime("%Y-%m-%d %H:%M:%S"))
        else: # 微调后模型默认覆盖此前模型，默认存放至checkpoint目录
            finetune_model_dir_after = finetune_model_dir_before # 也就是'checkpoint' 数据库中的model_file_path
            finetune_model_name_after = finetune_model_name_before

        # 更新training数据表status任务状态1
        sql_training_status = "UPDATE training SET status = 1, modified =%s WHERE id =%s"
        with dba as db:
            try:
                db.cursor.execute(sql_training_status, (str(time.strftime("%Y-%m-%d %H:%M:%S")), task_id))
                db.conn.commit()
            except:
                logger.error('数据库连接有误')
                logger.error(str(traceback.format_exc()))

        original_model_full_path = os.path.join(finetune_model_dir_before, finetune_model_name_before)
        new_model_full_path = os.path.join(finetune_model_dir_after, finetune_model_name_after)
        # 启动模型，开始训练
        domain_model_finetune(corpus_full_path, original_model_full_path, new_model_full_path)

        # 查看该任务当前状态，如果为4则已被用户取消
        sql_status = "SELECT id, status, monitoring_model_id FROM training WHERE id = %s"
        with dba as db:
            try:
                db.cursor.execute(sql_status,task_id)
                params = db.cursor.fetchall()
            except:
                logger.error('数据库连接有误')
                logger.error(str(traceback.format_exc()))

        status = params[0][1]
        monitoring_model_id = params[0][2]
        # 如果用户没有取消该训练任务，则正常更新model和training表
        if status != 4:
            # 更新model数据表，写入训练后模型数据
            sql_save_model = "UPDATE model SET model_size='368M', model_file_path=%s, model_file_name=%s, status=1, modified=%s WHERE id=%s"
            with dba as db:
                try:
                    db.cursor.execute(sql_save_model,(finetune_model_dir_after, finetune_model_name_after, str(time.strftime("%Y-%m-%d %H:%M:%S")), monitoring_model_id))
                    db.conn.commit()
                except:
                    logger.error('数据库连接有误')
                    logger.error(str(traceback.format_exc()))

            # 更新training数据表status任务状态为2已完成
            sql_training_status = "UPDATE training SET status = 2, modified =%s WHERE id =%s"
            with dba as db:
                try:
                    db.cursor.execute(sql_training_status,(str(time.strftime("%Y-%m-%d %H:%M:%S")),task_id))
                    db.conn.commit()
                except:
                    logger.error('数据库连接有误')

    except:
        traceback.print_exc()
        logger.error(sys.exc_info())
        # 更新training数据表status任务状态3
        sql_training_status = "UPDATE training SET status = 3, modified =%s WHERE id =%s"
        with dba as db:
            try:
                db.cursor.execute(sql_training_status, (str(time.strftime("%Y-%m-%d %H:%M:%S")), task_id))
                db.conn.commit()
            except:
                logger.error('数据库连接有误')

        # return json.dumps({"state_code": 400, "error_msg":traceback.format_exc()}, ensure_ascii=False)


class BertGenerate(Resource):

    def post(self):
        global p
        input_args = parser.parse_args()
        input_args = json.loads(json.dumps(input_args))
        input_tasks = json.loads(json.dumps(input_args['input_task']))

        # logger.debug('传入的组任务参数'+str(eval(input_tasks)).replace("'",'"'))
        for args in eval(input_tasks):
            # 检测是否已经完成生成数量

            with dba as db:
                sql = """SELECT num, progress, status FROM task WHERE id=%s"""
                try:
                    db.cursor.execute(sql,args['task_id'])
                    params = db.cursor.fetchall()
                except:
                    logger.error('数据库连接有误')

            params = params[0]
            number = params[0]
            progress = params[1]
            status = params[2]
            # 如果该任务id状态为4或已满足篇数，直接将status更新为4并返回
            if (status == 4) or (number <= progress):

                with dba as db:
                    sql = """UPDATE task SET status=4, modified=now() WHERE id=%s and num<=progress"""
                    try:
                        db.cursor.execute(sql, args['task_id'])
                        db.conn.commit()
                    except:
                        logger.error('数据库连接有误')

                return json.dumps({"state_code": 200}, ensure_ascii=False)
            elif (number > progress) and (status not in [2,5,6]):

                with dba as db:
                    sql = """UPDATE task SET status=1, modified=now() WHERE id=%s"""
                    try:
                        db.cursor.execute(sql, args['task_id'])
                        db.conn.commit()
                    except:
                        logger.error('数据库连接有误')

                # 如果属于组任务，则同时将组任务状态改为1
                if args["task_group_id"] != 0:

                    with dba as db:
                        sql = """UPDATE task SET status=1, modified=now() WHERE id=%s"""
                        try:
                            db.cursor.execute(sql, args['task_group_id'])
                            db.conn.commit()
                        except:
                            logger.error('数据库连接有误')
        p = Process(target=bert_gen, args=(input_tasks,))
        p.start()
        p.join()

class Finetune(Resource):
# 训练小模型，在具体方法内修改状态
    def post(self):
        global p
        input_args = parser.parse_args()
        input_task = input_args['input_task']
        args = json.loads(input_task.replace('\'','\"'))
        args = args[0]
        print(type(args))
        logger.info('输入模型参数: ' + str(args))
        p = Process(target=finetune, args=(args['task_id'],args['model_id'],args['corpus_full_path']))
        p.start()
        p.join()

class Gpt2Generate(Resource):

    def post(self):
        global p
        input_args = parser.parse_args()
        input_args = json.loads(json.dumps(input_args))
        input_tasks = json.loads(json.dumps(input_args['input_task']))
        # logger.debug('传入的组任务参数'+str(eval(input_tasks)).replace("'",'"'))
        for args in eval(input_tasks):

            # 检测是否已经完成生成数量
            with dba as db:
                sql = """SELECT num, progress, status FROM task WHERE id=%s"""
                try:
                    db.cursor.execute(sql, args['task_id'])
                    params = db.cursor.fetchall()
                except:
                    logger.error('数据库连接有误')

            params = params[0]
            number = params[0]
            progress = params[1]
            status = params[2]
            # 如果该任务id状态为4或已满足篇数，直接将status更新为4并返回
            if (status == 4) or (number <= progress):

                with dba as db:
                    sql = """UPDATE task SET status=4, modified=now() WHERE id=%s and num<=progress"""
                    try:
                        db.cursor.execute(sql, args['task_id'])
                        db.conn.commit()
                    except:
                        logger.error('数据库连接有误')

                return json.dumps({"state_code": 200}, ensure_ascii=False)

            elif (number>progress) and (status not in [2,5]):
                logger.debug('输入模型参数: ' + str(args['task_id'])+' '+str(args['title']))

                with dba as db:
                    sql = """UPDATE task SET status=1, modified=now() WHERE id=%s"""
                    try:
                        db.cursor.execute(sql, args['task_id'])
                        db.conn.commit()
                    except:
                        logger.error('数据库连接有误')

                # 如果属于组任务，则同时将组任务状态改为1
                if args["task_group_id"] != 0:

                    with dba as db:
                        sql = """UPDATE task SET status=1, modified=now() WHERE id=%s"""
                        try:
                            db.cursor.execute(sql, args['task_group_id'])
                            db.conn.commit()
                        except:
                            logger.error('数据库连接有误')

        p = Process(target=gpt2_gen, args=(input_tasks,))
        p.start()
        p.join()

api.add_resource(BertGenerate, '/bert-gen')
api.add_resource(Finetune, '/finetune')
api.add_resource(Gpt2Generate, '/gpt2-gen')

if __name__ == '__main__':

    app.debug = True
    app.run(host=base_args.host, port=base_args.port)