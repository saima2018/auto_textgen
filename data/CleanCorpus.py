# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@version:
@author: fzq
@contact: fanzhiqiang@bat100.net
@software: PyCharm
@file: CleanCorpos.py
@time: 2020/4/14 17:08
@description:对语料进行清洗，用于智能写作，目前支持的用户及语料类型如下：
1. 重庆医美领域，重庆医美领域还进行了部分后处理的工作，以删除不合规的部分语料
2. 青岛美容数据
"""
import csv
import os
import re
import sys

import xlrd

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from bs4 import BeautifulSoup
from tqdm import tqdm
import Levenshtein
from Utils.commonutils import IoUtils

# 需要删除的字符的正则
del_char_regex = re.compile(
    '【em55317ig】|www.*?com|http.*?com|www.*?cn|http.*?cn|[你您]好[,:!，： ！]|&#[0-9]+[;；]?|&nbsp[;；]+|O(∩_∩)O|~~|'
    '&[a-z]+|&#[0-9]+|健康解答：|[0-9]+_|\t|效果对比图！|楼主|\\.{7,}|【(.*)】|小编导读：|通过以上介绍，相信您已经有所了解。|c_4\\(\\);|'
    '\\.hzh {display: none; }|点击图片进入下一页|to\\(\\);|导读[：】]', re.I)

# 需要删除包含以下正则的句子
del_sent_regex = re.compile('400-[0-9]+-[0-9]+|欢迎咨询|致电|热线|提示.*在线客服|什么疑问|登录.*网|欢迎(在线|点击)咨询|'
                            '实习编辑|责任编辑|不得转载|请勿转载|谢绝转载|内容合作请联系|关注.*微信公众号|上一页\\d+|'
                            '\\d+下一页|显示全文|猜你喜欢：|图片来源：|友情链接|版权所有|公安机关备案号|或联网清理整顿|'
                            '本文来自|请与我们联系|三陪价码|年度盘点：|版权问题|推荐关闭浮层|右下浮动|二维码关注'
                            '覆盖底部推荐区的文字层级|随屏播按钮|padding-rightz|内容来自|免责声明|推荐阅读：|更多推荐：|'
                            '小编推荐：|系统推荐：|推荐信息：|微信.*咨询|公众号.*咨询|上一篇：|下一篇：|相关推荐阅读')

# 去除汉字间的空格，而不去除英文间的空格
space_regex = re.compile('[\u4e00-\u9fa5。\\.,，:：《》、\\(\\)（）？]? +(?<![a-zA-Z])|\\d+ +| +\\d+|[a-z A-Z]+')


def read_csv(file_in, out_list, add_title, cut_len):
    """
    读取csv文件，形成语料的list
    :param file_in: 输入的csv文件
    :param out_list: 输出的list
    :param add_title: 是否添加title
    :param cut_len: 文章的截取长度，如果文章长度大于这个值则保留，否则则删除
    :return: 共处理多少行数据，以及有效语料的条数
    """
    all_count = 0
    success_count = 0
    with open(file_in, 'r', encoding='utf-8')as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            all_count += 1
            if len(row) < 2:
                continue
            title = row[0].strip()
            content = row[1].strip()
            if len(content) < cut_len:
                continue
            content = combine_title_and_content(add_title=add_title, content=content, title=title)
            content = clean_content(content, cut_len)
            if content:
                out_list.append(content)
                success_count += 1
    return all_count, success_count


def read_excel(file_in, out_list, add_title, cut_len):
    """
    读取excel文件，形成语料的list
    :param file_in: 输入的csv文件
    :param out_list: 输出的list
    :param add_title: 是否添加title
    :param cut_len: 文章的截取长度，如果文章长度大于这个值则保留，否则则删除
    :return: 共处理多少行数据，以及有效语料的条数
    """
    workbook = xlrd.open_workbook(file_in)
    sheet = workbook.sheet_by_index(0)
    all_count = sheet.nrows - 1
    success_count = 0
    for i in tqdm(range(1, sheet.nrows)):
        # for i in range(1, sheet.nrows):
        title = sheet.row_values(i, 0)[0]
        if title == 'nike vandal high supreme复刻版上脚效果怎么样？':
            print('123')
        content = sheet.row_values(i, 1)[0].strip()
        if len(content) < cut_len:
            continue
        content = combine_title_and_content(add_title, content, title)
        content = clean_content(content, cut_len)
        if content:
            out_list.append(content)
            success_count += 1
    return all_count, success_count


def combine_title_and_content(add_title, content, title):
    """
    我们在生成文章时，通常是给定一个标题，所以标题比较重要。我们可以根据需要将标题加在正文前，组合形成语料
    :param add_title:
    :param content:
    :param title:
    :return:
    """
    if add_title:
        # 对title进行清洗
        if type(title) is not str:
            return content
        # title 去掉前后空格
        title = title.strip()
        if len(title) < 5:
            # 如果标题的字数少于5个，则不添加标题
            return content
        # "唯资丝绒哑光唇釉多少钱 唯资是什么品牌",这个标题时含有2个标题，但没有标点符号，所以要进行拆分出来
        tmp_match = re.search('[\u4e00-\u9fa5] [\u4e00-\u9fa5]', title)
        if tmp_match:
            title_0 = title[:tmp_match.start() + 1]
            title_1 = title[tmp_match.end():]
            title = add_title_symbol(title_0) + add_title_symbol(title_1)
        else:
            title = add_title_symbol(title)

        # 如果title 和content中前与相同字数的句子的编辑距离小于title的1/4，则说明重复了，返回content
        if Levenshtein.distance(title, content[:len(title)]) < 4:
            return content
        try:
            # 去掉content中与标题一模一样的句子，如果title中含有正则的特殊符号，则content不做处理
            content = re.sub('[。？！]' + title, '。', content)
            content = re.sub('^' + title, '', content)
        except Exception:
            pass
        return title + content
    else:
        return content


def add_title_symbol(title):
    if not title:
        return ''
    if title[-1] in ['？', '！', '。']:
        pass
    elif re.search('多|什么|嘛|吗|么|怎|不|哪|那|原因|如何|方法|价格|可以|好处', title):
        title += '?'
    else:
        title += '。'
    return title


def clean_content(text_in, cut_len):
    """
    处理内容
    :param text_in:
    :param cut_len: 文章的截取长度，如果文章长度大于这个值则保留，否则则删除
    :return:
    """
    # 如果没有一篇文章中没有中文，则不要这篇文章返回空
    if not re.search('[\u4e00-\u9fa5]', text_in):
        return ''
    # 通过BeautifulSoup去掉html相关标签
    soup = BeautifulSoup(text_in, 'html.parser')
    out_text = soup.get_text()
    # 去除文章开头前十位的空格之前的内容
    counter = 0
    for n in range(len(out_text)):
        counter += 1
        if out_text[n] == ' ':
            break
        if counter > 20:
            break
    if counter < 20:
        out_text = out_text[counter:]

    # 大于小于转换
    out_text = out_text.replace('&lt;', '<').replace('&gt;', '>').replace('大于大于', '大于')

    # 通过正则去除掉需要删除的字符
    out_text = re.sub(del_char_regex, '', out_text)
    # 全角转半角
    out_text = full_to_half(out_text)
    # 英文标点转中文标点
    out_text = eng_trans_to_chi(out_text)
    # 降温中的英文冒号修改为中文冒号
    out_text = replace_colon(out_text)
    # 如果文章中“问：答：”出现的次数超过4次，表明他是短句问答，不利于生成文章，需要舍弃
    if len(re.findall('问：|答：|Q[:：]|A[:：]', out_text)) > 2:
        return ''
    # 去除重复的字符
    if out_text:
        out_text = remove_duplicate_char(out_text)
    # 去除多余的空格
    out_text = remove_space(out_text)
    # 替换文本中部分.为句号
    if out_text:
        out_text = replace_dot(out_text)
    # 如果出现连续的标点符号，则用句号代替，如：RR；，；，`^ ；，；{，； {`被狂喜冲
    out_text = re.sub('[，。、；‘’【】{}：~！`^？]{2,}', '。', out_text)
    # 去除文章中不合适的句子
    if out_text:
        out_text = remove_invalid_article(out_text)
    if len(out_text) < cut_len:
        return ''
    return out_text


def full_to_half(ustring):
    """
    全角转半角
    :param ustring:
    :return:
    """
    ss = []
    for s in ustring:
        inside_code = ord(s)
        if inside_code == 12288:
            # 空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            # 其它按公式进行转换
            inside_code -= 65248
        ss.append(chr(inside_code))
    return ''.join(ss)


def eng_trans_to_chi(string):
    """
    将英文标点符号转换为中文标点符号
    :param string:
    :return:
    """
    e_pun = u',!?;'
    c_pun = u'，！？；'
    table = {ord(f): ord(t) for f, t in zip(e_pun, c_pun)}
    return string.translate(table)


def remove_duplicate_char(in_text):
    """
    去除文本中的重复字符
    :param in_text:输入的字符串，in_text='-----::普通'
    :return:
    """
    # 统计每个字符连续出现的次数,[{'-': 5}, {':': 2}, {'普': 1}, {'通': 1}]
    count_list = []
    count_char = in_text[0]
    count_json = {count_char: 1}
    for i in range(1, len(in_text)):
        if in_text[i] == count_char:
            count_json[count_char] += 1
        else:
            count_list.append(count_json)
            count_char = in_text[i]
            count_json = {count_char: 1}
    count_list.append(count_json)

    # 输出的字符列表
    out_list = []
    for tmp_jsoin in count_list:
        for k, v in tmp_jsoin.items():
            if v == 1:
                out_list.append(k)
            elif k in ['.'] or re.findall('[a-zA-Z]|[0-9]', k):
                out_list.append(k * v)
            elif k in ['。', '？', '—', '！', '，', '*', '；']:
                out_list.append(k)
            elif re.findall('[\u4e00-\u9fa5]', k):
                if v > 3:
                    out_list.append(k)
                else:
                    out_list.append(k * v)
            elif k in ['-']:
                out_list.append(k * 2)
            else:
                out_list.append(k)
    return ''.join(out_list)


def replace_dot(in_text):
    """
    替换句子中部分'.'为句号,英文和数字间的不替换
    :param in_text:
    :return:
    """
    # 判断文本结尾是否是一个汉字，如果是汉字则在末尾加上句号
    if re.search('[\u4e00-\u9fa5]', in_text[-1]):
        in_text += '。'
    # 替换句子中的◆|■|●|�|★|☆为句号
    in_text = re.sub('◆|■|●|◢|▲|�|★|▶|◎|◇|☆|↑|\r|\n', '。', in_text)
    txt_list = list(in_text)
    while True:
        tmp_match = re.search('[\u4e00-\u9fa5]\\.', in_text)
        if tmp_match:
            txt_list[tmp_match.start() + 1] = '。'
            in_text = ''.join(txt_list)
        else:
            return in_text


def replace_colon(in_text):
    # 将文章中汉字后的':'修改为'：'
    txt_list = list(in_text)
    while True:
        tmp_match = re.search('[\u4e00-\u9fa5]:', in_text)
        if tmp_match:
            txt_list[tmp_match.start() + 1] = '：'
            in_text = ''.join(txt_list)
        else:
            return in_text


def remove_space(in_text):
    """"
    处理中文之间的空格，不去除英文之间的空格
    """
    should_replace_list = space_regex.findall(in_text)
    order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        in_text = in_text.replace(i, new_i)
    return in_text.strip()


def remove_invalid_article(in_text):
    """
    删除掉广告等不需要的句子
    :param in_text:
    :return:
    """
    out_articles = []
    sentences = re.findall('.*?[。？！]', in_text)
    if len(sentences) < 2:
        return ''

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence[0] == '>':
            sentence = sentence[1:].strip()
        # 如果句子中仅含有数字加句号,类似“1。”,则替换为"1."
        if re.search('^\\d+。$', sentence):
            sentence = sentence[: -1] + '.'
            out_articles.append(sentence)
        # 或长句子重复，或句子中含有400电话等广告, 或单个句子长度大于150，或句子中不含中文，则舍弃这句话,
        elif len(sentence) < 3 or len(sentence) > 150 or (len(sentence) > 6 and sentence in out_articles) or re.search(
                del_sent_regex, sentence) or not re.search('[\u4e00-\u9fa5]', sentence):
            continue
        else:
            out_articles.append(sentence)
    return ''.join(out_articles)


def clean_corpus(dir_name, out_file, add_title, cut_len=128):
    """
    对原始的xlsx，xls，csv格式的文件进行读取，以及相关的清洗
    :param dir_name: 原始语料所在目录
    :param out_file: 清洗后的语料文件
    :param add_title: True 表示添加标题，False表示不添加标题
    :param cut_len: 文章的截取长度，如果文章长度大于这个值则保留，否则则删除
    :return:
    """
    result_list = []
    for root, _, files in os.walk(dir_name):
        for tmp_file in files:
            file_dir = os.path.join(root, tmp_file)
            print('处理文件：', file_dir)
            # 获取文件的扩展名
            file_ext_name = os.path.splitext(file_dir)[1]
            if file_ext_name in ['.xls', '.xlsx']:
                all_count, success_count = read_excel(file_dir, result_list, add_title, cut_len)
            elif file_ext_name == '.csv':
                # 由于csv格式经常会遇到编码格式不统一而导致读取错误的问题，所以统一要求客户上次excel格式的语料，
                # 如果确实有处理csv格式的需求再做处理
                continue
                # all_count, success_count = read_csv(file_dir, result_list, add_title, cut_len)
            else:
                # 其他格式的数据暂不处理，如txt格式等
                continue
            print('文件语料条数：{}，有效条数：{}，无效条数：{}\n'.format(all_count, success_count, all_count - success_count))

    print('=' * 50)
    print('语料长度', len(result_list))
    print('去重前语料条数；{}'.format(len(result_list)))
    result_list = list(set(result_list))
    print('去重后语料条数；{}'.format(len(result_list)))
    IoUtils.saveJson(result_list, out_file, out_format=True)


def post_clean_corpus(in_file):
    """
    对部分语料进行后处理，删除掉部分和领域无关的文章，如重庆客户提供的医美领域的文章含有很多其它疾病类的文章，进行后处理
    :param in_file:
    :return:
    """
    corpus_list = IoUtils.loadJson(in_file)
    err_count = 0
    for i in corpus_list:
        if re.search('道德经|劳动能力鉴定职工工伤与职业病致残等级|试题分析|花粉过敏|基本释义|狗狗|描写的|鼠标|白内障|'
                     '胚胎干细胞|肺癌|胸痛|直肠炎|眼镜蛇|脚气|视网膜|肿瘤|消化道|百病|痔疮|<p>|佛[说陀光手]|礼佛|'
                     '牙周炎|医改|磨牙症|止咳|咳嗽|小儿子|亲兄弟|牙龈疾病|伤残鉴定|电视剧|淘宝|痛风|操作系统', i):
            corpus_list.remove(i)
            err_count += 1
    IoUtils.saveJson(corpus_list, in_file, out_format=True)
    print('原始数据共：{}条，共去除不规范文章：{}条'.format(len(corpus_list), err_count))


def combine_corpus(text_1, text_2, out_file):
    """
    合并语料文件，有时候需要对几批语料进行组合，以形成新的语料
    :param text_1: 语料1
    :param text_2: 语料2
    :param out_file: 输出文件
    :return:
    """
    corpus_1 = IoUtils.loadJson(text_1)
    corpus_2 = IoUtils.loadJson(text_2)
    # 合并
    corpus_1.extend(corpus_2)
    # 去重
    corpus = list(set(corpus_1))
    # 保存
    IoUtils.saveJson(corpus, out_file, out_format=True)
    print('语料合并前：{}条，语料去重：{}条，语料合并后：{}条'.format(len(corpus_1), len(corpus_1) - len(corpus), len(corpus)))


if __name__ == '__main__':
    data_path = os.path.dirname(os.path.abspath(__file__))
    datasets_path = os.path.join(data_path, 'datasets')
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
        print('新建目录：', datasets_path)

    print(data_path)
    clean_corpus(add_title=False, cut_len=100, dir_name=os.path.join(datasets_path, '青岛_娱乐语料_0812'),
                 out_file=os.path.join(datasets_path, '青岛_娱乐_0814.json'))
    # clean_corpus(add_title=True, cut_len=200, dir_name=os.path.join(data_path, '重庆语料_2020-6-22'),
    #              out_file=os.path.join(datasets_path, 'version2_0628.json'))
    #
    # post_clean_corpus('datasets/version1_0628.json')
    #
    # combine_corpus(os.path.join(datasets_path, '青岛_美容_0716.json'), os.path.join(datasets_path, '青岛_美容_0717.json'),
    #                os.path.join(data_path, '青岛_美容add_all0718.json'))
