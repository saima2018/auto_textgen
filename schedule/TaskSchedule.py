# -*- coding: utf-8 -*-

import time
import threading
import requests
import os
import sys
import json
import math
import logging
import traceback

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from schedule.queuing_system import queuing, queue_group, queue_operations
from concurrent.futures import ThreadPoolExecutor
from Utils.DataUtils import load_xml_conf
from Utils.commonutils.MysqlUtils import dba
import Utils.logger_set as logger_set
# 设定日志文件
logger_set.setup_logging()
logger = logging.getLogger(__file__)

# 从配置文件中获取相应的参数
conf = load_xml_conf()
sub_task_size = conf['task_scheduler']['sub_task_batch_size']
gen_batch_size = conf['task_scheduler']['gen_batch_size']
gpu = conf['task_scheduler']['gpu']
gpu_num = conf['task_scheduler']['gpu_num']

# gpu的数量
gpu_count = len(gpu)
# 记录gpu的状态,全局变量，初始为不空闲
gpu_status_json = {}
for k in gpu.keys():
    gpu_status_json[k] = 1

# 初始化GPU生成锁，全局变量，初始为空闲
gen_lock = {}
gen_lock_counter = {} # 记录被锁定的次数，超过2次自动解锁
for k in gpu.keys():
    gen_lock[k] = 0
    gen_lock_counter[k] = 0

# 创建线程池来调用gpu，有多少个gpu则设置多少个线程
executor = ThreadPoolExecutor(max_workers=gpu_count)

# 主动拉取任务间隔时间，单位为秒
pull_task_int_time = 5
# 监测gpu是否有空闲间隔时间
sup_gpu_int_time = 5
# 当gpu状态为1时，检查使用率的间隔时间
gpu_status1_time = 5

def gpu_task_service(host, group_task, gpu_name):
    global gpu_status_json, gen_lock
    task = group_task[0]
    first_task_id = group_task[0]['task_id']
    last_task_id = group_task[-1]['task_id']
    # 任务类型：0表示生成，1表示训练
    task_type = int(task['task_type'])
    task_id = int(task['task_id'])
    headers = {
        "content-type": "application/json"
    }
    # 根据task_type和model_id将任务发送到对应的接口
    if int(task_type) == 0:
        if int(task['model_id']) == 1:
            # 大模型的model_id默认为1大模型生成
            url = host + '/gpt2-gen'
        else:
            # 如果是其他model_id都是使用小模型
            url = host + '/bert-gen'
    elif int(task_type) == 1:
        # 对模型进行训练
        url = host + '/finetune'
        task['task_group_id'] = 'NA'
    else:
        # 数据类型有误，放弃执行该操作
        logger.debug('task_type有误'+str(task_type))
        return

    logger.debug('当前线程号和接口：{}, {}, 当前子任务id范围: {}-{}, model_id: {}, operation: {}, task_group_id: {}'.
                 format(str(threading.currentThread().ident), str(url), str(first_task_id), str(last_task_id),
                        str(task['model_id']),str(task['operation']),str(task['task_group_id'])))


    input_args = {"input_task":str(group_task)}
    try:
        logger.info('发送post请求的时间：{}'.format(time.strftime("%H:%M:%S", time.localtime())))
        requests.post(url=url, headers=headers, data=json.dumps(input_args))
        logger.info('开锁时间：{}'.format(time.strftime("%H:%M:%S", time.localtime())))
        gen_lock[gpu_name] = 0
    except:
        logger.error("向{}发送任务报错".format(gpu_name))
        logger.error(str(traceback.format_exc()))


def schedule_task():
    global gpu_status_json, gen_lock
    while True:
        # 循环访问各个gpu
        for gpu_name, gpu_status in gpu_status_json.items():
            logger.info('GPU状态：{}, {}'.format(gpu_name, gpu_status))
            if gpu_status == 0:
                # 在gpu空闲时去数据库中拉取任务
                while True:
                    try: # 只在queue_group_list为空，且有gpu空闲时检测
                        # 等待5秒，防止组任务列表正在更新
                        time.sleep(pull_task_int_time)
                        # group_task_list = get_group_task_list()
                        group_task_list = queue_operations.read_queue_group_list()
                        if len(group_task_list) == 0:
                            # 只有所有gpu都是空闲状态时才检查异常暂停任务0、1、3，避免抓取（其他gpu）正在执行的（最后一个组合）任务
                            # gpu_memory = 0
                            # nvidia = os.popen('nvidia-smi | grep %').read().split('|')
                            # for n in range(math.floor(len(nvidia)/4)):
                            #     gpu_memory += int(nvidia[2+4*n].strip()[:-14])
                            # print('--------------------gpu memory sum',gpu_memory)
                            # if gpu_memory < 350:
                            logger.debug('进入异常暂停检测')
                            check_pause(120)
                            # 只在每分钟头5秒显示记录
                            if time.localtime().tm_sec in [0,1,2,3,4,5]:
                                logger.debug('未执行检测完成')
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('未执行检测超时')
                    try:
                        # group_task_list = get_group_task_list()
                        group_task_list = queue_operations.read_queue_group_list()
                        logger.debug('读到组任务列表'+str(len(group_task_list)))
                        if time.localtime().tm_sec in [0,1,2,3,4,5]:
                            logger.debug('任务队列检测完成')
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('ERROR任务队列检测超时')
                        group_task_list = []
                    if group_task_list:
                        # 如果拉取到任务则跳出该循环
                        break
                    else:
                        # 等待一定时间后再去拉取任务
                        time.sleep(pull_task_int_time)
                id_list = []
                for group_task_ in group_task_list:
                    for single_task_ in group_task_:
                        id_list.append(single_task_['task_id'])

                # logger.debug('当前全部子任务ID列表'+str(id_list))

                # 获取排在首位的任务，并修改数据库中的状态
                group_task = group_task_list.pop(0)
                # 组任务去重
                new_group_task = []
                id_ = []
                for i in group_task:
                    if i['task_id'] not in id_:
                        id_.append(i['task_id'])
                        new_group_task.append(i)
                group_task = new_group_task
                # 更新组任务列表
                try:
                    queue_operations.update_queue_group_list(str(group_task_list).replace('\'', '\"'))
                except:
                    logger.error(str(traceback.format_exc()))
                    logger.error('更新任务列表失败')
                #检查该task的id是否已完成
                group_task_list_catch = []

                for single_task in group_task:
                    if single_task['task_type'] == 0:
                        with dba as db:
                            sql = """SELECT num, progress, status FROM task WHERE id=%s"""
                            try:
                                db.cursor.execute(sql, single_task['task_id'])
                                params = db.cursor.fetchall()
                            except:
                                logger.error(str(traceback.format_exc()))
                                logger.error('数据库连接有误')

                        params = params[0]
                        number = params[0]
                        progress = params[1]
                        status = params[2]

                        if (status not in [2,4,5]) and (number>progress):
                            group_task_list_catch.append(single_task)
                    elif single_task['task_type'] == 1:
                        #  训练任务暂时不需要检查状态
                        group_task_list_catch.append(single_task)

                # 修改gpu状态并，提交任务给gpu对应的接口进行执行
                gpu_status_json[gpu_name] = 1 # 修改GPU状态为1

                logger.debug('{}生成锁状态(0为空闲，1为占用): {}'.format(gpu_name,gen_lock[gpu_name]))
                if gen_lock[gpu_name] == 0:
                    executor.submit(gpu_task_service, gpu[gpu_name]['host'], group_task_list_catch, gpu_name)
                    gen_lock[gpu_name] = 1
                    time.sleep(sup_gpu_int_time)
                else:

                    gen_lock_counter[gpu_name] += 1
                    logger.debug('{}生成锁锁定次数：{}，如果大于2次则自动解锁'.format(gpu_name,str(gen_lock_counter[gpu_name])))
                    # 锁定大于2次，自动解锁
                    if gen_lock_counter[gpu_name] >2:
                        gen_lock[gpu_name] = 0
                        gen_lock_counter[gpu_name] = 0
                        logger.debug('{}生成锁下次自动解锁'.format(gpu_name))

                    logger.debug('{}占用中，将该任务退还到任务列表，等待下一轮抓取'.format(gpu_name))
                    group_task_list.insert(0,group_task_list_catch)
                    queue_operations.update_queue_group_list(str(group_task_list).replace('\'', '\"'))
                    logger.info('更新后组任务列表长度: {}'.format(len(group_task_list)))



            # 如果GPU状态为1，需检查并排除任务没有正常返回200，使用率为0的情况
            elif gpu_status == 1:
                time.sleep(gpu_status1_time)
                gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
                for n in range(math.floor(len(gpu_status)/4)):
                    if gpu_num[gpu_name] == n:
                        gpu_util = gpu_status[3+4*n].strip()[:-13]
                if gpu_util in ['0%','1%','2%','3%','4%','5%','6%']:
                    time.sleep(gpu_status1_time)
                    nvidia_status = os.popen('nvidia-smi | grep %').read().split('|')
                    for n in range(math.floor(len(nvidia_status) / 4)):
                        if gpu_num[gpu_name] == n:
                            gpu_util = nvidia_status[3 + 4 * n].strip()[:-13]
                            memory_util = nvidia_status[2+4*n].strip()[:-14]
                    # if gpu_name == 'gpu1':
                    #     gpu_util = nvidia_status[3].strip()[:-13]
                    #     memory_util = nvidia_status[2].strip()[:-14]
                    # elif gpu_name == 'gpu2':
                    #     memory_util = nvidia_status[6].strip()[:-14]
                    #     gpu_util = nvidia_status[7].strip()[:-13]
                    # elif gpu_name == 'gpu3':
                    #     memory_util = nvidia_status[10].strip()[:-14]
                    #     gpu_util = nvidia_status[11].strip()[:-13]
                    # elif gpu_name == 'gpu4':
                    #     memory_util = nvidia_status[14].strip()[:-14]
                    #     gpu_util = nvidia_status[15].strip()[:-13]
                    # 如果显存占用小于350M,则认为没有任务在进行，将gpu状态改为0.
                    if (gpu_util in ['0%','1%','2%','3%','4%','5%','6%']) and (int(memory_util) < 350):
                        gpu_status_json[gpu_name] = 0

# 读取数据库中可能存在的因故未自动重启的执行中和系统暂停的生成任务，status1和3
# @timeout_decorator.timeout(15)
def check_pause(time_gap):
    # global current_task
    try:
        # 读取task表中处于0,1,3status的任务
        with dba as db:
            sql = "SELECT model_id, model_file_path, model_file_name, words, num, progress, keywords, group_id, title, \
                  immediately, id, is_group, status, sections, is_sub FROM task WHERE TIMESTAMPDIFF(HOUR, created, NOW())< %s \
                  AND status in (0,1,3) AND TIMESTAMPDIFF(MINUTE, modified, NOW())> 5"
            try:
                db.cursor.execute(sql, time_gap)
                all_params = db.cursor.fetchall()
            except:
                logger.error(str(traceback.format_exc()))
                logger.error('数据库连接有误')

        # 用于接收异常暂停的子任务，归并到一起
        group_queue_list = []
        for params in all_params:
            args = {}
            args["model_id"] = params[0]
            args["finetune_model_dir_after"] = params[1].decode('utf-8')
            args["finetune_model_name_after"] = params[2].decode('utf-8')
            args["words"] = params[3]
            args["num"] = params[4]
            args["progress"] = params[5]
            args["keywords"] = params[6]
            if args['keywords'] is None:
                args['keywords'] = ""
            args["task_group_id"] = params[7]
            args["title"] = params[8].decode('utf-8').replace('"','').replace("'",'') # 去掉引子中的英文单双引号
            args["immediately"] = params[9]
            args["task_id"] = params[10]
            args["task_type"] = 0
            args["operation"] = '006'
            args["is_group"] = params[11]
            args['status'] = params[12]
            args['paragraphs'] = params[13]
            # 将空值转为空字符串，以便排队时按json格式读取
            for key in args:
                if args[key] == None:
                    args[key] = ""
            # 如果篇数已满足，直接改status为4，同时将该id的任务全部清除
            if args['num']<= args['progress']:
                with dba as db:
                    sql = """UPDATE task SET status=4, modified=now() WHERE id=%s"""
                    try:
                        db.cursor.execute(sql, args['task_id'])
                        db.conn.commit()
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('数据库连接有误')

                queue_catch = []
                queue_now = queue_operations.read_queue_list()
                for task in queue_now:
                    if task['task_id'] != args['task_id']:
                        queue_catch.append(task)
                queue_operations.update_queue_list(str(queue_catch))
            else:
                # 直接找出子任务
                if (args['is_group'] != 1) and (args['status'] in [0,1,3]):
                    # 如果任务已处于queue队列，则不需要作为异常暂停读取
                    current_queue = queue_operations.read_queue_list()
                    # 生成任务001, 006
                    if len(current_queue) != 0:
                        for item in current_queue:
                            if (item["task_id"] == args["task_id"]) and (item['operation'] in ['001','006']): # 加operation排除同id的训练任务
                                logger.info("任务{}已存在".format(args["task_id"]))
                                # return "任务{}已存在".format(args["task_id"])

                    logger.debug('读到异常暂停任务, task_id: {}, model_id: {}, operation: {}, title: {}'.format(str(args['task_id']),
                                str(args['model_id']),str(args['operation']),str(args['title'])))
                    # 将该任务加入排队
                    group_queue_list.append(json.loads(json.dumps(args, cls=MyEncoder, indent=4)))

        logger.debug('异常暂停子任务个数 '+str(len(group_queue_list)))
        logger.debug('从check_pause进入queue_group方法')
        # group_queue_list.reverse() # 把group_queue_list逆序，保证异常暂停检测先进先出
        queue_group(group_queue_list)

    except:
        traceback.print_exc()
        if time.localtime().tm_sec in [0,1,2,3,4,5]:
            logger.debug('未读到异常任务')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    # 每次启动时将queue表初始化为空列表[]
    # update_task_list('[]')
    # update_group_task_list('[]')
    schedule_task()
