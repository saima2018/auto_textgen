#! /usr/lib python3.5
# -*- coding:utf-8 -*-
import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)
from flask import Flask
from flask_restful import Api, Resource, reqparse
from multiprocessing import Process
from Utils.DataUtils import load_xml_conf
from Utils.commonutils.MysqlUtils import *
from schedule.queuing_system import queuing, queue_group, queue_operations
from data.CleanCorpus import clean_corpus
import hashlib
import ast
import traceback
import requests
import time
import json
import logging
import Utils.logger_set as logger_set
# set up log
logger_set.setup_logging()
logger = logging.getLogger(__file__)

# load configuration
conf = load_xml_conf()
app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument("task_id", type=str, help='task id')
parser.add_argument("task_type", type=int, help='task type, 0 for generation, 1 for training')
parser.add_argument("operation", type=str, help='operation code')
parser.add_argument("request_time", type=int, help='timestamp in seconds')
parser.add_argument("sign", type=str, help='signature')

gpu = conf['task_scheduler']['gpu']
valid_seconds = conf['task_scheduler']['valid_seconds']

# number of GPUs
gpu_count = len(gpu)

# API encryption
def sign_body(body, apikey="apikey"):
    '''request body signature'''
    # generate a list of key=value formatted data
    a = ["".join(str(i[1])) for i in sorted(body.items()) if str(i[1]) and i[0] != "sign"]
    # order parameters in ascending order of ASCII code
    strA = "".join(a)
    # concatenate apikey to strA to get striSignTemp string
    striSignTemp = strA+apikey
    # print(striSignTemp.lower())

    # MD5 encryption
    def jiamimd5(src):
        m = hashlib.md5()
        m.update(src.encode('UTF-8'))
        return m.hexdigest()
    sign = jiamimd5(striSignTemp.lower())
    # add signature to body
    body["sign"] = sign
    # logging.info('Signature: ',sign)
    return body

class LoadParams(Resource):

    def post(self):
        input_args = parser.parse_args()
        args = parser.parse_args().copy()
        args = sign_body(args)
        logger.info('Input parameters: '+str(input_args))
        logger.info('Signed input '+str(args))
        # parameter check
        try:
            assert (round(time.time()) - input_args['request_time'] < valid_seconds) and \
                   (input_args['sign'] == args['sign'])
        except:
            logger.debug(str(traceback.format_exc()))
            return json.dumps({"state_code": 400, "error_msg": "Signature check failed"}, ensure_ascii=False)

        try:
            assert input_args['operation'] in ['001','002','003','004','005','006','007','101','102','103']
        except:
            logger.debug(str(traceback.format_exc()))
            return json.dumps({"state_code": 400, "error_msg": "Operation type error"}, ensure_ascii=False)

        try:
            assert input_args['task_type'] in [0, 1]
        except:
            logger.debug(str(traceback.format_exc()))
            return json.dumps({"state_code": 400, "error_msg": "Task type error"}, ensure_ascii=False)

        p = Process(target=extract_task, args=(args,))
        p.start()

        return json.dumps({"state_code": 200}, ensure_ascii=False)

# extract elements in task_id list and read corpus files
def extract_task(args):
    try:
        for id in ast.literal_eval(args['task_id']):
            args['task_id'] = id
            if args['task_type'] == 0:  # text generation task
                with dba as db:
                    sql = "SELECT model_id, model_file_path, model_file_name, words, num, progress, keywords, group_id, title, \
                                            immediately, sections, is_sub, is_group FROM task WHERE id=%s"
                    try:
                        db.cursor.execute(sql, str(args['task_id']))
                        params = db.cursor.fetchall()
                        params = params[0]
                        args['num'] = params[4]
                        args['progress'] = params[5]
                        args['is_sub'] = params[11]
                        args['is_group'] = params[12]
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('database connection error')
                # check if task is already finished
                if args['num'] <= args['progress']:
                    with dba as db:
                        sql_status = "UPDATE task SET status = 4, modified=now() WHERE id = %s"
                        try:
                            db.cursor.execute(sql_status, str(args['task_id']))
                            db.conn.commit()
                        except:
                            logger.error(str(traceback.format_exc()))
                            logger.error('database connection error')

                else:
                    # if backend sends a group task, then read each of the sub task in the group.
                    # the group_id of a sub task equals the task_id of its group task
                    # also check if sub task is finished

                    # group task will not enter the queuing method (for it takes too long),
                    # instead, it will be integrated into a list and passed into queue_group method
                    # and be updated to queue_group_list
                    if args['is_group'] == 1:
                        # if pause operation, delete sub tasks of the group task from two queue lists
                        if args['operation'] in ['002', '003', '004']:
                            # delete current task from queue list
                            current_queue = queue_operations.read_queue_list()
                            if len(current_queue) != 0:
                                queue_list = current_queue
                            else:
                                queue_list = []
                            queue_list_catch = []
                            for item in queue_list:
                                if item["task_group_id"] != args["task_id"]:
                                    queue_list_catch.append(item)
                            queue_list = queue_list_catch

                            queue_operations.update_queue_list(str(queue_list).replace('\'', '\"'))

                            logger.info('updated sub task list length: ' + str(len(queue_list)))

                            # remove this task from queue_group_list if present
                            try:
                                queue_group_list = queue_operations.read_queue_group_list()
                                queue_group_list_catch = []
                                for sub_list in queue_group_list:
                                    sub_list_catch = []
                                    try:
                                        for dict in sub_list:
                                            if dict['task_group_id'] != args["task_id"]:
                                                sub_list_catch.append(dict)
                                    except:
                                        pass
                                    if len(sub_list_catch) != 0:
                                        queue_group_list_catch.append(sub_list_catch)

                                queue_group_list = queue_group_list_catch
                                # update queue group list
                                queue_operations.update_queue_group_list(str(queue_group_list).replace('\'', '\"'))
                            except:
                                pass

                        # if not pause operation, read each sub task
                        else:
                            with dba as db:
                                sql = "SELECT model_id, model_file_path, model_file_name, words, num, progress, keywords, group_id, title,\
                                                    immediately, sections, id, status FROM task WHERE group_id=%s"
                                try:
                                    db.cursor.execute(sql, str(args['task_id']))
                                    group_params = db.cursor.fetchall()
                                except:
                                    logger.error(str(traceback.format_exc()))
                                    logger.error('db connection error')

                            group_queue_list = []
                            for n in range(len(group_params)):
                                params = group_params[n]
                                args["model_id"] = params[0]
                                args["finetune_model_dir_after"] = params[1].decode('utf-8')
                                args["finetune_model_name_after"] = params[2].decode('utf-8')
                                args["words"] = params[3]
                                args["num"] = params[4]
                                args["progress"] = params[5]
                                args["keywords"] = params[6]
                                if args['keywords'] is None:
                                    args['keywords']= ""
                                args["task_group_id"] = params[7]
                                args["title"] = params[8].decode('utf-8').replace('"', '').replace("'", "")
                                args["immediately"] = params[9]
                                args["paragraphs"] = params[10]
                                args['task_id'] = params[11]
                                args['status'] = params[12]
                                if (args['num'] > args['progress']) and (args['status'] not in [2,4,5]):
                                    # turn null value to empty string for json operation
                                    for key in args:
                                        if args[key] == None:
                                            args[key] = ""
                                    # logger.info('parameters sent to queue method' + str(args).replace('\'', '\"'))

                                    # 如果传来的是组任务，直接将所有任务加入group_queue_list列表，再将词列表整体更新到数据库，以节约时间
                                    # if group task, add all tasks to group_queue_list, and update in database to save time
                                    group_queue_list.append(json.loads(json.dumps(args, cls=MyEncoder, indent=4)))
                                # if task is finished, change status code to 4
                                elif args['num'] <= args['progress']:

                                    with dba as db:
                                        sql_status = "UPDATE task SET status = 4, modified=now() WHERE id = %s"
                                        try:
                                            db.cursor.execute(sql_status, str(args['task_id']))
                                            db.conn.commit()
                                        except:
                                            logger.error(str(traceback.format_exc()))
                                            logger.error('数据库连接有误')

                            logger.info('更新后子任务列表长度: ' + str(len(group_queue_list)))
                            # 以下两行专为组生成任务，其他单个任务的参数args已加入queuing方法处理
                            # queue_list = get_task_list()   不能在此读取，因为任务大时，上面的更新还没成功，造成读取到空列表
                            logger.debug('从extract_task进入queue_group方法')
                            queue_group(group_queue_list)

                    # if not group task
                    elif (args['is_group'] != 1) and (args['is_sub'] !=1):
                        args["model_id"] = params[0]
                        args["finetune_model_dir_after"] = params[1].decode('utf-8')
                        args["finetune_model_name_after"] = params[2].decode('utf-8')
                        args["words"] = params[3]
                        args["num"] = params[4]
                        args["progress"] = params[5]
                        args["keywords"] = params[6]
                        if args['keywords'] is None:
                            args['keywords'] = ""
                        else:
                            args["keywords"] = params[6].decode('utf-8')
                        args["task_group_id"] = params[7]
                        args["title"] = params[8].decode('utf-8').replace('"', '').replace("'", "")
                        args["immediately"] = params[9]
                        args["paragraphs"] = params[10]
                        args["is_sub"] = params[11] # 是否属于某个组任务
                        if args['num'] > args['progress']:
                            # 将空值转为空字符串，以便排队时按json格式读取
                            for key in args:
                                if args[key] == None:
                                    args[key] = ""
                            # logger.info('传入排队函数的参数' + str(args))
                            queuing(json.loads(str(args).replace('\'', '\"')))
                            current_queue_list = queue_operations.read_queue_list()
                            queue_group(current_queue_list)

            elif args['task_type'] == 1:  # 训练任务
                args['title'] = "NA" # 增加一个title值，便于queue_group方法统一排序
                with dba as db:
                    sql = "SELECT model_id, corpus_url, corpus_filename FROM training WHERE id=%s"
                    try:
                        db.cursor.execute(sql, str(args['task_id']))
                        params = db.cursor.fetchall()
                        params = params[0]
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('数据库连接有误')

                args["model_id"] = params[0]
                # 训练任务不能使用通用大模型
                if args['model_id'] == 1:
                    raise Exception('训练任务不能使用通用大模型')
                    # return json.dumps({"state_code": 400, "error_msg": '训练任务不能使用通用大模型'}, ensure_ascii=False)
                args['corpus_url'] = str(params[1],encoding='utf-8')
                filename = str(params[2],encoding='utf-8')
                # # 读取训练文本url，并转存为本地txt文件
                url = args['corpus_url']

                try:
                    response = requests.get(url)
                except:
                    return json.dumps({"state_code": 400, "error_msg": '读取语料文件失败'}, ensure_ascii=False)

                args['corpus_full_path'] = os.path.join(root_path, 'data', filename)
                with open(args['corpus_full_path'],'wb') as local_excel:
                    local_excel.write(response.content)
                # clean_corpus(filename)
                # filename = filename[:-5] + '.json'

                # 将空值转为空字符串，以便排队时按json格式读取
                for key in args:
                    if args[key] == None:
                        args[key] = ""
                logger.info('传入排队函数的参数' + str(args))
                # 直接更新至组合任务列表
                if args['operation'] == '101':
                    args_list = []
                    args_list.append(json.loads(str(args).replace('\'', '\"')))
                    print(args_list,'===========')
                    queue_operations.update_queue_group_list([args_list])
                else: # 直接从组合任务列表删除
                    queue_group_list = queue_operations.read_queue_group_list()
                    queue_group_list_catch = []
                    for queue_group in queue_group_list:
                        if queue_group[0]['task_type'] == 1:
                            if queue_group[0]['task_id'] == args['task_id']:
                                pass
                            else:
                                queue_group_list_catch.append(queue_group)
                        else:
                            queue_group_list_catch.append(queue_group)
                    queue_operations.update_queue_group_list(queue_group_list_catch)

    except:
        logger.error(str(traceback.format_exc()))
        logger.error('Task extraction error')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)

api.add_resource(LoadParams, '/')
if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0', port='1092')

