# -*- coding:utf8 -*-

import math
import sys
import ast
import traceback
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import logging
logger = logging.getLogger(__file__)
from Utils.commonutils.MysqlUtils import *
from Utils.DataUtils import load_xml_conf

# 从配置文件中获取子任务批量大小
conf = load_xml_conf()
sub_task_batch_size = conf['task_scheduler']['sub_task_batch_size']
gen_batch_size = conf['task_scheduler']['gen_batch_size']
gpu = conf['task_scheduler']['gpu']
gpu_count = len(gpu)

def queuing(task_args, sub_task_batch_size=sub_task_batch_size):
    """
    该函数的输入为单个子任务，输出queue_list列表,同时会更新queue_list队列文件
    :param task_args:
    :param sub_task_batch_size:
    :return:
    """
    queue_list = []
    logger.info('新任务ID:{},模型ID:{},操作类型:{},引子：{}'.format(str(task_args['task_id']),
            str(task_args['model_id']),str(task_args['operation']),str(task_args['title'])))

    # 已生成篇数检验
    if task_args['task_type'] == 0:
        if task_args['num']<=task_args['progress']:

            with dba as db:
                sql = """UPDATE task SET status=4, modified=now() WHERE id=%s and num<=progress"""
                try:
                    db.cursor.execute(sql, task_args['task_id'])
                    db.conn.commit()
                except:
                    logger.error(str(traceback.format_exc()))
                    logger.error('数据库连接有误')

            logger.debug('Task_id{}生成文章数已满足需要，不再继续生成'.format(task_args['task_id']))
            # return '生成文章数已满足需要，不再继续生成'

    # 002暂停等待中的生成,003删除等待中的生成,004暂停进行中的生成,102删除等待中的训练,103删除进行中的训练
    # 需要按优先级排队的生成、训练任务列表，包含operation 001新增生成,101新增训练,006恢复生成
    # 从数据库读取 queue_list

    # current_queue = get_task_list()
    current_queue = queue_operations.read_queue_list()
    if len(current_queue) != 0:
        # current_queue = json.loads(str(current_queue).replace("'",'"'))
        queue_list=current_queue

    logger.info("更新前子任务列表长度"+str(len(queue_list)))
    # 更新前子任务id和operation列表
    id_list = []
    for task in queue_list:
        id_list.append('ID_' + str(task['task_id']))
        id_list.append('OP_' + str(task['operation']))
    logger.info('更新前所有任务ID和operation' + str(id_list))

    # 生成任务001, 006
    if task_args["operation"] in ["001","006"]:
        # 确认task_id是新增
        if len(queue_list) != 0:
            for item in queue_list:
                if (item["task_id"] == task_args["task_id"]) and (item['operation'] == task_args['operation']):
                    logger.info("任务{}已存在".format(task_args["task_id"]))
                    return "任务{}已存在".format(task_args["task_id"])

        queue_list.append(task_args)

        # 立即修改即时生成任务的状态为进行中

        if task_args["immediately"] == 2:

            with dba as db:
                sql_status = "UPDATE task SET status = 1, modified=now() WHERE id = %s"
                try:
                    db.cursor.execute(sql_status, task_args['task_id'])
                    db.conn.commit()
                except:
                    logger.error(str(traceback.format_exc()))
                    logger.error('数据库连接有误')

                    # 如果属于组任务，则同时将组任务状态改为1
            if task_args["task_group_id"] !=0:

                with dba as db:
                    sql_status = "UPDATE task SET status = 1, modified=now() WHERE id = %s"
                    try:
                        db.cursor.execute(sql_status, task_args['task_group_id'])
                        db.conn.commit()
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('数据库连接有误')

    # 002暂停等待中的生成，003删除等待中的生成，004暂停进行中的生成
    if task_args["operation"] in ["002", "003", "004"]:
        # 从queue_list中去除该任务
        queue_list_catch = []
        for item in queue_list:
            if item["task_id"] != task_args["task_id"]:
                queue_list_catch.append(item)
        queue_list = queue_list_catch

        with dba as db:
            sql_status = "UPDATE task SET status = 2, modified=now() WHERE id = %s"
            try:
                db.cursor.execute(sql_status, task_args['task_id'])
                db.conn.commit()
            except:
                logger.error(str(traceback.format_exc()))
                logger.error('数据库连接有误')

        # 从queue_group_list中去除该任务, 如果组合任务列表中没有该任务，则不执行操作
        try:
            queue_group_list = queue_operations.read_queue_group_list()
            queue_group_list_catch = []
            for sub_list in queue_group_list:
                sub_list_catch = []
                for dict in sub_list:
                    if dict['task_id'] != task_args["task_id"]:
                        sub_list_catch.append(dict)
                queue_group_list_catch.append(sub_list_catch)
            queue_group_list = queue_group_list_catch
            # 更新组合任务列表
            queue_operations.update_queue_group_list(str(queue_group_list).replace('\'', '\"'))

        except:
            pass

        with dba as db:
            sql_status = "UPDATE task SET status = 2, modified=now() WHERE id = %s"
            try:
                db.cursor.execute(sql_status, task_args['task_id'])
                db.conn.commit()
            except:
                logger.error(str(traceback.format_exc()))
                logger.error('数据库连接有误')

        # 如果属于组任务，则将组任务状态改为暂停2或5,组任务只会批量暂停
        if task_args["task_group_id"] != 0:
            if task_args["operation"] in ["002", "004"]:

                with dba as db:
                    sql_status = "UPDATE task SET status = 2, modified=now() WHERE id = %s"
                    try:
                        db.cursor.execute(sql_status, task_args['task_group_id'])
                        db.conn.commit()
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('数据库连接有误')

            elif task_args["operation"] in ["003"]:

                with dba as db:
                    sql_status = "UPDATE task SET status = 5, modified=now() WHERE id = %s"
                    try:
                        db.cursor.execute(sql_status, task_args['task_group_id'])
                        db.conn.commit()
                    except:
                        logger.error(str(traceback.format_exc()))
                        logger.error('数据库连接有误')

    # 新增训练任务101，直接
    if task_args["operation"] == "101":
        # 确认task_id是新增
        if len(queue_list) != 0:
            for item in queue_list:
                if (item["task_id"] == task_args["task_id"]) and (item['operation'] == task_args['operation']):
                    logger.info("任务{}已存在".format(task_args["task_id"]))
                    # return "任务{}已存在".format(task_args["task_id"])
        task_args["immediately"] = 1
        queue_list.append(task_args)
    # 删除等待中的训练102，从队列中按task_id去除
    if task_args["operation"] == '102':
        queue_list_catch = []
        for item in queue_list:
            if item["task_id"] != task_args["task_id"]:
                queue_list_catch.append(item)
        queue_list = queue_list_catch

        with dba as db:
            sql_status = "UPDATE training SET status = 4, modified=now() WHERE id = %s"
            try:
                db.cursor.execute(sql_status, task_args['task_id'])
                db.conn.commit()
            except:
                logger.error(str(traceback.format_exc()))
                logger.error('数据库连接有误')

    # 如果operation是删除进行中的训练，则不更改队列
    if task_args["operation"] == "103":
        pass

    # 按immediately的值进行排序
    queue_list = sorted(queue_list, key=lambda x:(x["immediately"]), reverse=True)

    queue_operations.update_queue_list(str(queue_list).replace('\'','\"'))

    logger.info('更新后子任务列表长度: ' + str(len(queue_list)))
    # 更新后子任务id和operation列表
    updated_id_list = []
    for task in queue_list:
        updated_id_list.append('ID_'+str(task['task_id']))
        updated_id_list.append('OP_'+str(task['operation']))

    # logger.info('更新后所有任务ID和operation: ' + str(updated_id_list)+'  ')

    return queue_list

def queue_group(queue_list):
    """
    组合queue_list中的单个生成任务，相同模型和优先级的任务合并到组合列表，每个组合列表内生成总数在20~40之间，并将组合列表存入queue_group_list
    :param queue_list: queue_list 单个任务列表
    :return: 无
    """
    logger.debug('进入queue_group方法，对子任务进行组合，并更新到queue_group_list.txt')
    # 按immediately的值和title长度进行排序
    # queue_list = sorted(queue_list, key=lambda x: (x["immediately"],len(x['title'])), reverse=True)

    # 子任务从queue_list组合入queue_group_list, 即将它从queue_list去除
    queue_list_db = queue_operations.read_queue_list()
    queue_list_db_catch = []
    for single_task in queue_list_db:
        if single_task not in queue_list:
            queue_list_db_catch.append(single_task)
    queue_operations.update_queue_list(str(queue_list_db_catch).replace('\'','\"'))
    queue_group_list = queue_operations.read_queue_group_list() # 读取当前已有的组合任务列表

    # 将训练任务直接加入任务组合列表,随即将该训练任务剔除出queue_list
    queue_group_training = []  # 保证queue_group_list格式正确，列表套列表
    queue_list_catch_non_training = []
    for single_task in queue_list:
        if single_task['operation'] in ['101']:
            queue_group_training.append(single_task)
        else:
            queue_list_catch_non_training.append(single_task) # 如果不是训练任务，则保留
    if len(queue_group_training) != 0:
        queue_group_list.insert(0, queue_group_training)
    queue_list = queue_list_catch_non_training # 此时queue_list只有生成任务

    model_id_list = []
    for single_task in queue_list:
        if (single_task["operation"] in ["001", "006"]) and (single_task['model_id'] not in model_id_list):
            model_id_list.append(single_task['model_id'])
    logger.info('model id list: '+str(model_id_list))
    for model_id in model_id_list:
        num_counter = 0
        for single_task in queue_list:
            # single_task = json.loads(str(single_task).replace("'",'"'))
            if (single_task['operation'] in ['001', '006']) and (single_task['model_id'] == model_id):
                num_counter += single_task['num']
        logger.info('公用模型子任务个数：'+str(num_counter)+' 模型ID: '+str(model_id))
        # 如果counter总数不到100，则将所有任务合并为一个组合
        if num_counter <= 100:
            queue_group_1 = []
            for single_task in queue_list:
                if (single_task['operation'] in ['001', '006']) and (single_task['model_id'] == model_id):
                    queue_group_1.append(single_task)
            # 如果新加入的组合任务列表优先级为1，则添加到总组合任务列表最左边
            if queue_group_1[0]['immediately'] == 1:
                queue_group_list.insert(0,queue_group_1)
            else:
                queue_group_list.append(queue_group_1)

        # 如果counter除以gpu数大于60，则将所有任务合并为若干num总数不超过60的组合
        elif num_counter / gpu_count >= 200:
            queue_group_2 = []
            group_num_counter = 0
            for single_task in queue_list:
                if (single_task['operation'] in ['001', '006']) and (single_task['model_id'] == model_id):
                    group_num_counter += single_task['num']
                    if group_num_counter < 200:
                        queue_group_2.append(single_task)
                    else:
                        if queue_group_2[0]['immediately'] == 1:
                            queue_group_list.insert(0, queue_group_2)
                        else:
                            queue_group_list.append(queue_group_2)

                        group_num_counter = single_task['num']
                        queue_group_2 = []
                        queue_group_2.append(single_task)
            # 如果新加入的组合任务列表优先级为1，则添加到总组合任务列表最左边
            if queue_group_2[0]['immediately'] == 1:
                queue_group_list.insert(0, queue_group_2)
            else:
                queue_group_list.append(queue_group_2)

        # 其他情况，直接将counter除以gpu数，生成gpu数个组合任务
        else:
            cutoff = math.ceil(num_counter/gpu_count)
            queue_group_3 = []
            group_num_counter = 0
            for single_task in queue_list:
                if (single_task['operation'] in ['001', '006']) and (single_task['model_id'] == model_id):
                    group_num_counter += single_task['num']
                    if group_num_counter < cutoff+2:
                        queue_group_3.append(single_task)
                    else:
                        if queue_group_3[0]['immediately'] == 1:
                            queue_group_list.insert(0, queue_group_3)
                        else:
                            queue_group_list.append(queue_group_3)

                        group_num_counter = single_task['num']
                        queue_group_3 = []
                        queue_group_3.append(single_task)
            # 如果新加入的组合任务列表优先级为1，则添加到总组合任务列表最左边
            if queue_group_3[0]['immediately'] == 1:
                queue_group_list.insert(0, queue_group_3)
            else:
                queue_group_list.append(queue_group_3)

    # 更新任务组合列表
    queue_operations.update_queue_group_list(str(queue_group_list).replace('\'', '\"'))

    logger.info('更新后组任务列表长度: ' + str(len(queue_group_list)))

class Queue_list_operations():
    def read_queue_list(self):
        """
        从queue.txt中获取子任务列表
        :return: queue_list子任务列表
        """
        # queue_list初始必须为[],创建的txt为ANSI编码，eval()转为list
        try:
            with open(os.path.join(root_path, 'schedule/queue_list.txt'), 'r', encoding='utf-8') as f:
                result = f.readline()
            queue_list = eval(result)
        except:
            logger.error(str(traceback.format_exc()))
            logger.error("读取子任务列表报错")
            queue_list = []
        finally:
            f.close()
        return queue_list

    def read_queue_group_list(self):
        """
        从queue_group_list.txt中获取子任务列表
        :return: queue_group_list子任务列表
        """
        # queue_list初始必须为[]
        try:
            with open(os.path.join(root_path, 'schedule/queue_group_list.txt'), 'r', encoding='utf-8') as f:
                result = f.readline()
            queue_group_list = eval(result)
        except:
            logger.error(str(traceback.format_exc()))
            logger.error("读取组合任务列表报错")
            queue_group_list = []
        finally:
            f.close()
        return queue_group_list

    def update_queue_list(self, new_queue_list):
        """
        更新任务子队列到queue.txt中
        :param new_value:
        :return:
        """
        new_queue_list = str(new_queue_list)

        try:
            with open(os.path.join(root_path, 'schedule/queue_list.txt'), 'w', encoding='utf-8') as f:
                f.write(new_queue_list)
        except:
            logger.error(str(traceback.format_exc()))
            logger.error('更新子任务列表报错')
        finally:
            f.close()

        logger.info("子任务列表更新成功，当前排队子任务数为：{}".format(len(eval(new_queue_list))))

    def update_queue_group_list(self, new_queue_group_list):
        """
        更新组合任务队列到queue_group_list.txt
        :param new_value:
        :return:
        """
        new_queue_group_list = str(new_queue_group_list)
        try:
            with open(os.path.join(root_path, 'schedule/queue_group_list.txt'), 'w', encoding='utf-8') as f:
                f.write(new_queue_group_list)
        except:
            logger.error(str(traceback.format_exc()))
            logger.error('更新组合任务列表报错')
        finally:
            f.close()

        logger.info("组合任务列表更新成功，当前排队组合任务数为：{}".format(len(eval(new_queue_group_list))))

queue_operations = Queue_list_operations()

if __name__ == '__main__':
    # dba = DBAction()
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
