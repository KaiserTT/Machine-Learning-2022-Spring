import sys
import pdb
import pprint
import logging
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace

import re

import sentencepiece as spm

data_dir = './DATA/rawdata'
dataset_name = 'ted2020'

prefix = Path(data_dir).absolute() / dataset_name

src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'


def strQ2B(ustring):
    # 负责将全角字符（比如中文字符或全角标点）转换成半角字符
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:
                # 全角空格（Unicode码为12288）半角空格（ASCII码为32）
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                # 检查字符是否为其他全角字符
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def clean_s(s, lang):
    # 用于清理单个字符串
    # 对于英文（en），它会删除括号内的文本、去除连字符、保留并标准化标点符号
    # 对于中文（zh），它会将全角字符转换成半角字符、删除括号内的文本、去除连字符、保留并标准化标点符号
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)  # 使用正则表达式去除括号及其内部的内容
        s = s.replace('-', '')  # 去除连字符
        s = re.sub('([.,;!?()\"])', r' \1 ', s)  # 标准化标点符号, 使其周围有空格
    elif lang == 'zh':
        s = strQ2B(s)  # 将全角字符转换为半角字符
        s = re.sub(r"\([^()]*\)", "", s)
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)
    s = ' '.join(s.strip().split())  # 去除字符串首尾的空白字符，并确保单词间只有一个空格
    return s


def len_s(s, lang):
    # 用于计算单个字符串的长度
    if lang == 'zh':
        return len(s)
    return len(s.split())


def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    # 用于清理语料库
    # 先检查清理后的文件是否已存在
    # 如果不存在，它逐行读取两种语言的语料文件，使用clean_s函数进行清理
    # 并根据设定的长度和长度比例标准（ratio, max_len, min_len）筛选句子
    # 最后，它将清理和筛选后的语句写入新的文件中。
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}', encoding='utf-8').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r', encoding='utf-8') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r', encoding='utf-8') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w', encoding='utf-8') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w', encoding='utf-8') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0:  # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0:  # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0:  # remove by ratio of length
                            if s1_len / s2_len > ratio or s2_len / s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

valid_ratio = 0.01  # 3000~4000 would suffice
train_ratio = 1 - valid_ratio

if (prefix / f'train.clean.{src_lang}').exists() \
        and (prefix / f'train.clean.{tgt_lang}').exists() \
        and (prefix / f'valid.clean.{src_lang}').exists() \
        and (prefix / f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')
else:
    # 生成标签作为数据集中句子的索引, 并随机打乱这些标签
    line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}', encoding='utf-8'))
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w', encoding='utf-8')
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w', encoding='utf-8')
        count = 0
        for line in open(f'{data_prefix}.clean.{lang}', 'r', encoding='utf-8'):
            # 根据判预设的训练集比例断当前行是否应该归入训练集
            if labels[count] / line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()

# 训练 SentencePiece 模型并使用其进行分词, 将句子转换成一系列子词 subword tokenization
import sentencepiece as spm

vocab_size = 8000
if (prefix / f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix / f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,  # 设定字符覆盖率为 100%
        model_type='unigram',  # 选择模型类型为 'unigram', 也可以选择 'bpe'
        input_sentence_size=1e6,  # 指定用于训练的句子数量, 设定每次读入的句子数量为 100w
        shuffle_input_sentence=True,  # 在训练前对句子进行随机排序
        normalization_rule_name='nmt_nfkc_cf',  # 使用 'nmt_nfkc_cf' 来规范化文本
    )

spm_model = spm.SentencePieceProcessor(model_file=str(prefix / f'spm{vocab_size}.model'))
in_tag = {
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix / f'{split}.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix / f'{split}.{lang}', 'w', encoding='utf-8') as out_f:
                with open(prefix / f'{in_tag[split]}.{lang}', 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)


