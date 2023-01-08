# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@version:
@author: fzq
@contact: fanzhiqiang@bat100.net
@software: PyCharm
@file: base_domain_model_bert_train.py
@time: 2020/7/23 15:54
@description: 训练基础领域模型，要求单词训练的语料要达到500G，训练好基础模型后，以后每增加100M语料可在此基础上进行微调。
微调3次后需要将所有数据整合到一起，然后再重新训练基础模型。
注意：每次训练必须修改的参数如下：
1.语料地址：corpus_path
2.模型地址：model_path，模型的命名规则建议“客户公司名_领域名_语料日期_训练轮次_训练日期.model”
3.方法just_show中的三个文章标题，如果更换领域需要进行更改为相关的标题
4.os.environ["CUDA_VISIBLE_DEVICES"]：指定一块GPU进行训练，如果不指定则默认使用第0块

可选择修改的参数：
epochs：训练轮次，从经验来看可以设置为1000，大概需要5天时间。不同领域可能会做部分调整。

"""
import os
import re
import sys
import time
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from Utils.commonutils import IoUtils

# 指定选用gpu显卡
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# tensorflow日志级别，只显示error的日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

maxlen = 256
batch_size = 12
steps_per_epoch = 1000

epochs = 800

corpus_path = os.path.join(root_path, 'data', 'datasets', '青岛_娱乐_0814.json')
bert_path = os.path.join(root_path, 'models', 'chinese_L-12_H-768_A-12')
model_path = os.path.join(root_path, 'models', 'model_corpus_青岛娱乐0814_epoch1000.weights')
print('corpus_path:', corpus_path)

config_path = os.path.join(bert_path, 'bert_config.json')
checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
dict_path = os.path.join(bert_path, 'vocab.txt')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startwith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

articles = []
corpus_list = IoUtils.loadJson(corpus_path)
for article in corpus_list:
    article = article.replace(u'\u3000', ' ')
    sents = []
    for t in article.split('  '):
        for s in re.findall(u'.*?。', t):
            if len(s) <= maxlen - 2:
                sents.append(s)
    articles.append(sents)

data = []
pbar = tqdm(desc=u'构建语料中', total=sum(len(n) for n in articles))
print('articles length:{}'.format(len(articles)))
for tmp_article in articles:
    s = u''
    for i in range(len(tmp_article)):
        for j in range(len(tmp_article) - i):
            if len(s) + len(tmp_article[i + j]) > maxlen - 2:
                data.append(s)
                s = u''
                break
            else:
                s += tmp_article[i + j]
        pbar.update(1)
        if i + j >= len(tmp_article):
            break
    if s:
        data.append(s)

pbar.close()
np.random.shuffle(data)


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


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)
model.load_weights('/home/ubuntu/masai/textgen/textgen_lc_bert/models/model_corpus_青岛娱乐0814_epoch1000.weights')
# model.summary()
# 交叉熵作为loss，并mask掉输入部分的预测
y_true = model.input[0][:, 1:]  # 目标tokens
y_mask = model.get_layer('Embedding-Token').output_mask[:, 1:]  # 目标mask
y_mask = K.cast(y_mask, K.floatx())  # 转为浮点型
y_pred = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


class ArticleCompletion(AutoRegressiveDecoder):
    """
    基于随机采样的文章续写
    """

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids = inputs[0]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.zeros_like(token_ids)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=40):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample_1([token_ids[:-1]], n, topk)  # 基于随机采样
        return [text + tokenizer.decode(ids) for ids in results]


article_completion = ArticleCompletion(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def just_show():
    s1 = u'最新香港娱乐新闻曝光'
    s2 = u'蔡徐坤现身篮球场'
    s3 = u'王俊凯易烊千玺粉丝大打出手'
    for s in [s1, s2, s3]:
        t = article_completion.generate(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % ('\n'.join(t)))


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(model_path)
        # 演示效果
        # if epoch % 50 == 0:
            # model.save(os.path.join(root_path, 'models', str(epoch),'.h5' ))
            # model.save_weights(os.path.join(root_path,'models','青岛娱乐',str(epoch)+'.weights'))
        just_show()


if __name__ == '__main__':
    evaluator = Evaluate()
    train_generator = data_generator(data, batch_size)
    start_time = time.time()
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )
    end_time = time.time()
    hour = (end_time - start_time) / 3600
    print('训练{}轮花费时间：{}h'.format(epochs, hour))
