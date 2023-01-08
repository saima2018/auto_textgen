# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@version:
@author: fzq
@contact: fanzhiqiang@bat100.net
@software: PyCharm
@file: bert_gen.py
@time: 2020/4/27 12:26
@description:
"""

import json
import os
import re
import sys
import time
import traceback
import numpy as np
import math
import random
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from schedule.queuing_system import queue_operations
from Utils.commonutils.MysqlUtils import *
from Utils.DataUtils import load_xml_conf
from tqdm import tqdm
from data.CleanCorpus import clean_content
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
import logging
import Utils.logger_set as logger_set
# 设定日志文件
logger_set.setup_logging()
logger = logging.getLogger(__file__)
# 从配置文件中获取指定GPU序号的参数
conf = load_xml_conf()
patch_sentences = conf['patch']['list']
top_p = conf['task_scheduler']['top_p']
top_k = conf['task_scheduler']['top_k']

# 向生成的文本中插入关键词
def insert_keywords(text,keywords):
    try:
        if len(keywords)!= 0:
            keywords_list = keywords.split(';')
            # 如果文本中已经包含关键词，则从关键词表中去除
            keywords_list_catch = []
            for word in keywords_list:
                if word not in text:
                    keywords_list_catch.append(word)
            keywords_list = keywords_list_catch
            number_keywords = len(keywords_list)
            period_list = [k for k in range(len(text)) if text[k] in ['。' , '！' ,'？','，']]
            len_periods = len(period_list)
            period_index_interval =(1/(number_keywords+1))*len_periods
            text_splits = {}
            for i in range(number_keywords):
                text_index_previous = period_list[int(period_index_interval*i)]
                text_index = period_list[int(period_index_interval*(i+1))]
                text_splits[i] = text[text_index_previous+1:text_index+1]
                if i == number_keywords-1:
                    text_splits[number_keywords] = text[text_index+1:]
            patch_list = patch_sentences
            result = []
            for n in range(number_keywords):
                text_splits[n] = text_splits[n] + keywords_list[n] + patch_list[random.randint(0,len(patch_list)-1)]
                result.append(text_splits[n])
            result.append(text_splits[number_keywords])
            result = ''.join(result).replace('\n', '')
            for i in range(len(result) - 1, 0, -1):
                if result[i] in ['。', '！', '？','，']:
                    result = result[:i + 1]
                    break
            result = result[:-1] + '。'
            return result
        else:
            for i in range(len(text) - 1, 0, -1):
                if text[i] in ['。', '！', '？','，',',',';','；']:
                    text = text[:i + 1]
                    break
            result = text[:-1] + '。'
            return result
    except:
        logger.error('加入关键词失败')
        logger.error(str(traceback.format_exc()))
        return text

# 将文章分段,加上<p>标签
def make_paragraphs(text, paragraphs=1):
    try:
        period_list = [k for k in range(len(text)) if text[k] in ['。' , '！' ,'？','，']]
        len_periods = len(period_list)
        # 默认分三段
        period_index_interval =math.floor((1/paragraphs)*len_periods)
        text_splits = {}
        for i in range(paragraphs):

            text_index_previous = period_list[int(period_index_interval*i)-1]
            text_index = period_list[int(period_index_interval*(i+1))-1]
            text_splits[i] = '<p>'+ text[text_index_previous+1:text_index]+'。</p>'
            if i == 0:
                text_splits[i] = '<p>'+ text[0:text_index]+'。</p>'
            if i == paragraphs-1:
                text_splits[paragraphs] = text[text_index + 1:]
            # print(text_splits[i])
        result = []
        for n in range(paragraphs):
            result.append(text_splits[n])
        result = ''.join(result).replace('\n', '')
        return result
    except:
        return text

# 文章循环重复检测，如果文本中有任意连续五个字出现了五次及以上，则弃用此文本
def text_quality_check(text):
    quartet_list = []
    for n in range(len(text)-5):
        quartet_list.append(text[n:n + 5])
    for quartet in quartet_list:
        count = quartet_list.count(quartet)
        if count > 4:
            return None
    return text

def remove_repeat(article_in):
    """
    以句号为单位，删除文章中重复的句子,如果最后一句没有写完，则删除最后一句
    :param article_in:
    :return:
    """
    sent_list = list(filter(None, re.split('[。！]',article_in)))
    print(sent_list)
    # 如果最后一个字符不是句号，则说明句子不完整，将列表的最后一个元素设置为空
    if article_in[-1] != '。':
        sent_list[-1] = ''
    # 因为去空函数将最后一个表示句号的空值去掉了，所以要在最后加一个空值
    if sent_list[-1]:
        sent_list.append('')
    if len(sent_list) <= 1:
        return article_in
    out_list = []
    for i in sent_list:
        if len(i) >= 5: # 每个句号分隔段长度须大于2
            if i not in out_list:
                out_list.append(i)
        # elif len(i) < 10:
        #     out_list.append(i)
    return '。'.join(out_list) + '。'

maxlen = 256
batch_size = 16
steps_per_epoch = 200
epochs = 10
model_batch = 10

# bert模型文件、模型参数配置以及词典文件
bert_path = os.path.join(root_path, 'models', 'chinese_L-12_H-768_A-12')
config_path = os.path.join(bert_path, 'bert_config.json')
checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
dict_path = os.path.join(bert_path, 'vocab.txt')

# def bert_gen(task_id, task_group_id, model_id, title, num, model_path, model_name, max_length,paragraphs=1, topk=20, topp=None, keywords=None):
def bert_gen(input_args):

    input_args = eval(input_args)
    # 组任务共用参数
    model_id = input_args[0]['model_id']
    model_path = input_args[0]['finetune_model_dir_after']
    model_name = input_args[0]['finetune_model_name_after']
    # task_group_id = input_args[0]['task_group_id']
    num = input_args[0]['num']
    try:
        max_length = input_args[0]['words'] + 20  # 预先加20字，弥补可能的正则处理损失
    except:
        max_length = 500


    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )

    tokenizer = Tokenizer(token_dict, do_lower_case=True)
    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='lm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    class TextGen(AutoRegressiveDecoder):
        """基于随机采样的文本生成"""

        @AutoRegressiveDecoder.set_rtype('probas')
        def predict(self, inputs, output_ids, step):
            # token_ids = inputs[0]
            if output_ids.shape == (1, 0):
                output_ids = np.repeat(output_ids, inputs.shape[0], axis=0)
            token_ids = np.concatenate([inputs, output_ids], 1)

            # 单个序列的长度
            seq_length = token_ids.shape[1]
            if seq_length < 250:
                pass
            else:
                # 为防止序列过长后偏离主题，每生成200字，为接下来生成的10个字添加title的信息
                for n in range(1, 10):
                    if 200 * n <= seq_length < 210 * n:
                        token_ids = np.concatenate([inputs, token_ids[:, -100:]], axis=1)
                else:
                    token_ids = token_ids[:, -100:]

            segment_ids = np.zeros_like(token_ids)
            pred_start = time.time()
            tmp_out = model.predict([token_ids, segment_ids])[:, -1]
            return tmp_out

        def generate(self, title_list, n=1, topk=20, topp=None):
            tmp_list = []
            # 标题最大长度
            title_list = [title.replace(' ', '') for title in title_list]
            max_token_ids = 0
            for title in title_list:
                token_ids, _ = tokenizer.encode(title)
                len_token_ids = len(token_ids)
                if len_token_ids > max_token_ids:
                    max_token_ids = len_token_ids
            for title in title_list:
                # 如果该标题的字符数不是最大，则在前面添加[pad]
                token_ids, _ = tokenizer.encode(title)
                # 去掉最后的终止符
                token_ids = token_ids[:-1]
                for i in range(max_token_ids - len(token_ids)):
                    token_ids.insert(1, 0)
                    # token_ids.append(0)
                tmp_list.append(token_ids)
            results = self.random_sample(tmp_list, n, topk, topp)  # 基于随机采样
            articles = [tokenizer.decode(ids) for ids in results]
            out_articles = []
            for i in range(len(title_list)):
                # out_article = clean_article(article)
                out_article = title_list[i] + articles[i]
                # 写数据库
                out_articles.append(out_article)
            return out_articles

    article_completion = TextGen(
        start_id=None,
        end_id=tokenizer._token_end_id,
        maxlen=max_length
    )
    # 加载模型权重
    model_full_path = os.path.join(root_path, model_path, model_name)
    model.load_weights(model_full_path)
    # 从生成的批量文章中依次处理和存入每篇文章，每次model_batch篇
    for n in range(math.floor(len(input_args)/model_batch)+1):
        # 新建空白批量引子列表，作为generate方法的输入
        title_list = []
        for args in input_args[model_batch*n:model_batch*(n+1)]:
            # 子任务各自参数
            try:
                title = args['title']
            except:
                title = '输入标题参数有误，请忽略这篇文章'
            title_list.append(title)
        logger.info(title_list)
        start_time = time.time()
        try:
            result_list = article_completion.generate(title_list, n=num)
        except:
            logger.error('批量生成文章报错')
            logger.error(str(traceback.format_exc()))

        logger.info('time used: {}'.format(time.time()-start_time))
        counter = 0
        for args in input_args[model_batch*n:model_batch*(n+1)]:
            # num = args['num']
            task_id = args['task_id']
            task_group_id = args['task_group_id']
            keywords = args['keywords']
            paragraphs = args['paragraphs']
            result = result_list[counter]
            counter += 1
            if keywords:
                result = insert_keywords(result, keywords)
            result =remove_repeat(result)
            # 如果实际长度小于要求的20%，则放弃存入本篇文章，待重新生成
            # if result != None:
            #     if len(result) < max_length*0.2:
            #         continue
            result = make_paragraphs(result, paragraphs) # 分段要在清洗后面，否则<p>会被删除

            # result是单篇完整的文章，可以在这里加入添加到数据库
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
                # 如果篇数未满足要求，且未被暂停未完成，则将文章写入数据库
                if (number > progress) and (status not in [2,4,5,6]):

                    with dba as db:
                        query = """INSERT INTO article (task_id, model_id, task_group_id, content, status, created, modified) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
                        tuple = (task_id, model_id, task_group_id, result, 0, str(time.strftime("%Y-%m-%d %H:%M:%S")),
                                 str(time.strftime("%Y-%m-%d %H:%M:%S")))
                        try:
                            db.cursor.execute(query, tuple)
                            db.conn.commit()
                        except:
                            logger.error(str(traceback.format_exc()))
                            logger.error('数据库连接有误')

                    logger.info('文章已写入数据库，任务ID：{}， 文章长度：{}'.format(str(task_id),len(result)))

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
                    if (number <= progress) and (status not in [2,4,5,6]):

                        with dba as db:
                            sql_status = "UPDATE task SET status = 4, modified=now() WHERE id = %s"
                            try:
                                db.cursor.execute(sql_status, task_id)
                                db.conn.commit()
                            except:
                                logger.error(str(traceback.format_exc()))
                                logger.error('数据库连接有误')

                        # 如果属于组任务，则同时将已完成的组任务状态改为4
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

                elif status in [2,5,6]:
                    logger.info('该任务被暂停或取消：'+str(task_id))
                    # 同时去除队列中属于同一父任务的任务
                    # 从queue_group_list中去除该任务, 如果组合任务列表中没有该任务，则不执行操作
                    try:
                        # queue_group_list = get_group_task_list()
                        queue_group_list = queue_operations.read_queue_group_list()
                        queue_group_list_catch = []
                        for sub_list in queue_group_list:
                            sub_list_catch = []
                            try:
                                for dict in sub_list:
                                    if dict['task_group_id'] != task_group_id:
                                        sub_list_catch.append(dict)
                            except:
                                logger.debug(str(traceback.format_exc()))
                            if len(sub_list_catch) != 0:
                                queue_group_list_catch.append(sub_list_catch)

                        queue_group_list = queue_group_list_catch
                        # 更新组合任务列表
                        queue_operations.update_queue_group_list(str(queue_group_list))
                    except:
                        logger.debug(str(traceback.format_exc()))

                    break
                elif number<=progress:
                    logger.info('文章数量已满足生成要求：'+str(task_id))
            except:
                logger.error('文章写入数据库失败：'+str(task_id))
                logger.error(str(traceback.format_exc()))

        else:
            continue
        break # 跳出最外层for循环，结束组任务
    # return 0

def remove_repeat(article_in):
    """
    以句号为单位，删除文章中重复的句子,如果最后一句没有写完，则删除最后一句
    :param article_in:
    :return:
    """
    sent_list = list(filter(None, article_in.split('。')))
    # 如果最后一个字符不是句号，则说明句子不完整，将列表的最后一个元素设置为空
    if article_in[-1] != '。':
        sent_list[-1] = ''
    # 因为去空函数将最后一个表示句号的空值去掉了，所以要在最后加一个空值
    if sent_list[-1]:
        sent_list.append('')
    if len(sent_list) <= 1:
        return article_in
    out_list = []
    for i in sent_list:
        if i not in out_list:
            out_list.append(i)
        elif len(i) < 10:
            out_list.append(i)
    return '。'.join(out_list)

def domain_model_finetune(corpus_path=None, ori_model_path=None, new_model_path=None):
    articles = []
    corpus_list = json.load(open(corpus_path, 'r', encoding='utf-8'))
    for txt in corpus_list:
        txt = txt.replace(u'\u3000', ' ')
        sents = []
        for t in txt.split('  '):
            for s in re.findall(u'.*?。', t):
                if len(s) <= maxlen - 2:
                    sents.append(s)
        articles.append(sents)
    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)
    data = []
    pbar = tqdm(desc=u'构建语料中', total=sum(len(n) for n in articles))
    for article in articles:
        s = u''
        for i in range(len(article)):
            for j in range(len(article) - i):
                if len(s) + len(article[i + j]) > maxlen - 2:
                    data.append(s)
                    s = u''
                    break
                else:
                    s += article[i + j]
            pbar.update(1)
            if i + j >= len(article):
                break
        if s:
            data.append(s)
    pbar.close()
    np.random.shuffle(data)

    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='lm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )
    # 交叉熵作为loss，并mask掉输入部分的预测
    y_true = model.input[0][:, 1:]  # 目标tokens
    y_mask = model.get_layer('Embedding-Token').output_mask[:, 1:]  # 目标mask
    y_mask = K.cast(y_mask, K.floatx())  # 转为浮点型
    y_pred = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(1e-5))
    if ori_model_path:
        model.load_weights(ori_model_path)

    class data_generator(DataGenerator):
        """
        数据生成器
        """

        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids = [], []
            for is_end, text in self.sample(random):
                token_ids, segment_ids = tokenizer.encode(text)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    yield [batch_token_ids, batch_segment_ids], None
                    batch_token_ids, batch_segment_ids = [], []

    class StoryCompletion(AutoRegressiveDecoder):
        """
        基于随机采样的文章续写
        """

        @AutoRegressiveDecoder.set_rtype('probas')
        def predict(self, inputs, output_ids, step):
            token_ids = inputs[0]
            token_ids = np.concatenate([token_ids, output_ids], 1)
            segment_ids = np.zeros_like(token_ids)
            return model.predict([token_ids, segment_ids])[:, -1]

        def generate(self, text, n=1, topk=5):
            token_ids, _ = tokenizer.encode(text)
            results = self.random_sample([token_ids[:-1]], n, topk)  # 基于随机采样
            return [text + tokenizer.decode(ids) for ids in results]

    class Evaluate(keras.callbacks.Callback):
        """
        评估函数，保存最优模型并演示效果
        """

        def __init__(self):
            self.lowest = 1e10

        def on_epoch_end(self, epoch, logs=None):
            # 保存最优
            if logs['loss'] <= self.lowest:
                self.lowest = logs['loss']
                model.save_weights(new_model_path)
            # 演示效果
            just_show()

    article_completion = StoryCompletion(
        start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
    )

    def just_show():
        s1 = u'假体隆胸手术后怎么护理效果好？'
        s2 = u'全瓷牙与烤瓷牙有何区别？'
        s3 = u'激光祛斑的能量高会不会破坏皮肤？'
        for s in [s1, s2, s3]:
            t = article_completion.generate(s)
            print(u'输入: %s' % s)
            print(u'结果: %s\n' % ('\n'.join(t)))

    evaluator = Evaluate()
    train_generator = data_generator(data, batch_size)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )


if __name__ == '__main__':
    # domain_model_fintuen(corpus_path=os.path.join(root_path, 'data', 'datasets', '10M.json'),
    #                      ori_model_path=os.path.join(root_path, 'models', 'best_model_all_epoch_500_0416.weights'),
    #                      new_model_path=os.path.join(root_path, 'models', 'fintune_model_0427.weights'))

    # s1 = bert_gen(title="士大夫是稍等", num=2,
    #                  model_path=os.path.join(root_path, 'models'),model_name='best_model_all_epoch_500_0416.weights',
    #                  max_length=800, topk=25,task_group_id=1,task_id=1,model_id=1,paragraphs=1)
    # print(s1)
    # pass
    a='<p>脸上长痤疮了，怎么办？我们通常说的痤疮主要是由于长痘、内分泌失调引起的一种皮肤疾病。那么长痘、内分泌失调到底是什么呢？脸上长痘、内分泌失调到底是什么呢？下面教大家怎么帮助我们解决痘痘哦！皮肤长痘痘主要是由于内分泌失调引起的皮肤过敏，通常在多汗、情绪抑郁、压力、抑郁的时候比较容易长痘。一、内分泌失调形成的原因内分泌失调是有关生活方式的常见问题，但是这些问题往往和过去的黑色素有关。但是这些问题往往和过去的黑色素有关。但是这些问题往往和过去的黑色素有关。内分泌失调的原因可能会直接或间接的引起皮肤屏障功能失调，肌肤缺乏维生素b12，导致皮肤细胞代谢不顺利，也就是让有害物质存在于皮肤中。经常长痘痘的女性群体常见的问题是内分泌失调。脸上长痘痘的原因形成的问题可能是自己的内分泌失调，因为分泌失调而导致皮肤失去水分，缺乏水分的保护，就会引起皮肤的干燥、脱皮。内分泌失调的肌肤如果保养不当的话，还会引起皮肤过敏。二、如何有效祛痘当你还不知道祛痘是什么，就开始寻找祛痘方法，那么就真的是不负责任的去保养。而且不知道怎么护肤，怎么用护肤品都不知道怎么美白。想去缩小毛孔，又没有处理干净，那么就真的是灾难了。其实毛孔中的黑色素是最容易爬墙的地方。</p>'
    print(remove_repeat(a))

