import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from flask import Flask
from flask_restful import Api, Resource, reqparse
import re
from generate.bert_gen import insert_keywords, make_paragraphs, text_quality_check
from data.CleanCorpus import clean_content
import tensorflow as tf
import numpy as np
import traceback
import time
from schedule.queuing_system import queue_operations
from Utils.commonutils.MysqlUtils import *
from Utils.DataUtils import load_xml_conf
from train.modeling import GroverModel, GroverConfig, sample
from tokenization import tokenization
import logging
import Utils.logger_set as logger_set
# 设定日志文件
logger_set.setup_logging()
logger = logging.getLogger(__file__)

# 从配置文件中获取指定GPU序号的参数
conf = load_xml_conf()
patch_sentences = conf['patch']['list']

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # 不使用gpu

##### ignore tf deprecated warning temporarily
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
#####


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('metadata_fn', type=str, help='Path to a JSONL containing metadata')
parser.add_argument('out_fn', type=str, help='Out json, which will contain the completed jsons')
parser.add_argument('title', type=str, help='Text to complete', )  # 前端
parser.add_argument('keywords', type=str, help='关键词字符串，用;分隔', )  # 前端
parser.add_argument('model_config_fn', default='models/5.2G/configs/mega.json', type=str, help='Configuration JSON for the model')
parser.add_argument('cn_big_model', default='models/5.2G/model.ckpt-100000', type=str, help='checkpoint file for the model')  # 前端
parser.add_argument('batch_size', default=1, type=int, help='How many things to generate per context. will split into chunks if need be')  # 前端
parser.add_argument('num_folds', default=1, type=int, help='Number of folds. useful if we want to split up a big file into multiple jobs.')
parser.add_argument('max_batch_size', default=None, type=int, help='max batch size. You can leave this out and we will infer one based on the number of hidden layers')
parser.add_argument('top_p', type=float, help="for top p sampling. if this isn't none, use this for everthing")  # 前端
parser.add_argument('words', default=500, type=int, help='min length of sample')  # 前端
parser.add_argument('eos_token', default=60000, type=int, help='eos token id')
parser.add_argument('num', default=5, type=int, help='生成文章篇数;如果使用了batch_size,则num须能被batch_size整除。')
parser.add_argument('timestamp', type=int)  # 前端
parser.add_argument('sign', default='', type=str)  # 前端
parser.add_argument('return_as_list', default=False)
parser.add_argument('model_id')
parser.add_argument('task_id')
parser.add_argument('task_group_id')
parser.add_argument('restore_from', default='latest', help='whether to train from fresh model or latest checkpoint')  # 前端

def extract_generated_target(output_tokens, tokenizer):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_ind = 0
    end_ind = output_tokens.shape[0]

    return {
        'extraction': tokenization.printable_text(''.join(tokenizer.convert_ids_to_tokens(output_tokens))),
        'start_ind': start_ind,
        'end_ind': end_ind,
    }

# GPT2大模型生成
# def gpt2_gen(model_config_fn,max_batch_size,batch_size,top_p,cn_big_model,title,num,eos_token_input,words,keywords,
#                task_id,model_id,task_group_id,paragraphs):
def gpt2_gen(input_args):
    # 转换为列表类型
    input_args = eval(input_args)
    # 组任务共用参数
    model_id = input_args[0]['model_id']
    model_config_fn = 'models/5.2G/configs/mega.json'
    max_batch_size = None
    batch_size = 1
    top_p = 0.9
    cn_big_model = 'models/5.2G/model.ckpt-100000'
    eos_token_input = 60000
    task_group_id = input_args[0]['task_group_id']
    # 如果输入字数参数有误，则自动生成600字文章
    try:
        words = input_args[0]['words']
    except:
        words = 600
    proj_root_path = os.path.dirname(os.path.realpath(__file__))
    vocab_file_path = os.path.join(root_path, "tokenization/bert-base-chinese-vocab.txt")
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)
    news_config = GroverConfig.from_json_file(model_config_fn)

    # We might have to split the batch into multiple chunks if the batch size is too large
    default_mbs = {12: 32, 24: 16, 48: 3}
    max_batch_size = max_batch_size if max_batch_size is not None else default_mbs[
        news_config.num_hidden_layers]

    # factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
    num_chunks = int(np.ceil(batch_size / max_batch_size))
    batch_size_per_chunk = int(np.ceil(batch_size / num_chunks))

    # This controls the top p for each generation.
    top_p_now = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * top_p

    tf_config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
        p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
        eos_token = tf.placeholder(tf.int32, [])
        min_len = tf.placeholder(tf.int32, [])
        tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                               eos_token=eos_token, min_len=min_len, ignore_ids=None, p_for_topp=p_for_topp,
                               do_topk=False)

        saver = tf.train.Saver()
        saver.restore(sess, cn_big_model)
        # print('🍺Model loaded. \nInput something please:⬇️')
        # text = input()
        for args in input_args:
            # 子任务各自参数
            task_id = args['task_id']
            try:
                title = args['title']
            except:
                title = '输入标题参数有误，自动生成无效文本'
            num = args['num']
            keywords = args['keywords']
            paragraphs = args['paragraphs']
            text = title
            # while text != "":
            for i in range(num):
                print("Sample,", i + 1, " of ", num)
                line = tokenization.convert_to_unicode(text)
                bert_tokens = tokenizer.tokenize(line)
                encoded = tokenizer.convert_tokens_to_ids(bert_tokens)
                context_formatted = []
                context_formatted.extend(encoded)
                # Format context end
                gens = []
                gens_raw = []
                gen_probs = []
                for chunk_i in range(num_chunks):
                    tokens_out, probs_out = sess.run([tokens, probs],
                                                     feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                                eos_token: eos_token_input,
                                                                min_len: words,
                                                                p_for_topp: top_p_now[chunk_i]})

                    for t_i, p_i in zip(tokens_out, probs_out):
                        extraction = extract_generated_target(output_tokens=t_i, tokenizer=tokenizer)
                        gens.append(extraction['extraction'])
                result = re.findall('.{1,70}', gens[0].replace('[UNK]', '').replace('##', ''))
                result = '\n'.join(result)
                result = result[:words]
                # 插入关键词
                result = insert_keywords(result, keywords)
                # 检查循环重复
                # result = clean_content(result)
                # 分段
                result = make_paragraphs(result, paragraphs)
                if result is not None:
                    print('==========================\n',result)
                    try:
                        # 检测是否已经完成生成数量

                        with dba as db:
                            sql = """SELECT num, progress, status FROM task WHERE id=%s"""
                            try:
                                db.cursor.execute(sql, task_id)
                                params = db.cursor.fetchall()
                            except:
                                logger.error(str(traceback.format_exc()))
                                logger.error('数据库连接有误')

                        params = params[0]
                        number = params[0]
                        progress = params[1]
                        status = params[2]
                        if (number > progress) and (status not in [2,4,5]):

                            with dba as db:
                                query = """INSERT INTO article (task_id, model_id, task_group_id, content, status, created, modified) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
                                tuple = (
                                task_id, model_id, task_group_id, result, 0, str(time.strftime("%Y-%m-%d %H:%M:%S")),
                                str(time.strftime("%Y-%m-%d %H:%M:%S")))
                                try:
                                    db.cursor.execute(query, tuple)
                                    db.conn.commit()
                                except:
                                    logger.error(str(traceback.format_exc()))
                                    logger.error('数据库连接有误')

                            logger.info('文章已写入数据库' + str(task_id))

                            # 检测是否已经完成生成数量
                            with dba as db:
                                sql = """SELECT num, progress, status FROM task WHERE id=%s"""
                                try:
                                    db.cursor.execute(sql, task_id)
                                    params = db.cursor.fetchall()
                                except:
                                    logger.error(str(traceback.format_exc()))
                                    logger.error('数据库连接有误')

                            params = params[0]
                            number = params[0]
                            progress = params[1]
                            status = params[2]
                            if (number <= progress) and (status not in [2,4,5]):

                                with dba as db:
                                    sql_status = "UPDATE task SET status = 4, modified=now() WHERE id = %s"
                                    try:
                                        db.cursor.execute(sql_status, task_id)
                                        db.conn.commit()
                                    except:
                                        logger.error(str(traceback.format_exc()))
                                        logger.error('数据库连接有误')

                                # 同时清空queue表中可能剩余的该子任务
                                queue_list_catch = []
                                # current_queue = get_task_list()
                                current_queue = queue_operations.read_queue_list()

                                for item in current_queue:
                                    if (item["task_id"] != task_id) and (item['immediately'] != 1):  # 不等于1则不是训练任务
                                        queue_list_catch.append(item)
                                queue_list = queue_list_catch
                                # update_task_list(queue_list)
                                queue_operations.update_queue_list(str(queue_list))
                                # 如果属于组任务，则读取组任务进度，将已完成的组任务状态改为4
                                if task_group_id != 0:
                                    # 检测是否已经完成生成数量
                                    with dba as db:
                                        sql = """SELECT num, progress, status FROM task WHERE id=%s"""
                                        try:
                                            db.cursor.execute(sql, task_group_id)
                                            params = db.cursor.fetchall()
                                            params = params[0]
                                            number = params[0]
                                            progress = params[1]
                                            status = params[2]
                                        except:
                                            logger.error(str(traceback.format_exc()))
                                            logger.error('数据库连接有误')

                                    if number <= progress:
                                        with dba as db:
                                            sql_status = "UPDATE task SET status = 4, modified=now() WHERE id = %s"
                                            try:
                                                db.cursor.execute(sql_status, task_group_id)
                                                db.conn.commit()
                                            except:
                                                logger.error(str(traceback.format_exc()))
                                                logger.error('数据库连接有误')

                        elif status in [2, 5, 6]:
                            logger.info('该任务被暂停或取消：' + str(task_id))
                            break
                        elif number <= progress:
                            logger.info('文章数量已满足生成要求：' + str(task_id))
                    except:
                        traceback.print_exc()
                        logger.debug('文章写入数据库失败：' + str(task_id))
                else:
                    pass

            else:
                continue
            break #跳出外层循环，结束该组任务
