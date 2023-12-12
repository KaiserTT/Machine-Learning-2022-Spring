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
from fairseq import utils
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)


# 定义 RNN 编码器
# RNNEncoder: 对于每个输入标记, 编码器将生成一个输出向量和一个隐藏状态向量, 并且隐藏状态向量传递到下一步
class RNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens

        self.embed_dim = args.encoder_embed_dim  # 设置嵌入维度
        self.hidden_dim = args.encoder_ffn_embed_dim  # 设置隐藏层维度
        self.num_layers = args.encoder_layers  # 设置 RNN 的层数
        # 定义输入层的 dropout 层
        self.dropout_in_module = nn.Dropout(args.dropout)
        # 创建一个 GRU (门控循环单元)网络，设置为双向
        self.rnn = nn.GRU(
            self.embed_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=args.dropout,
            batch_first=False,
            bidirectional=True
        )
        # 定义输出层的 dropout 层
        self.dropout_out_module = nn.Dropout(args.dropout)
        # 获取填充 token 的索引
        self.padding_idx = dictionary.pad()

    # 在双向循环神经网络(Bi-directional Recurrent Neural Network, Bi-RNN)中
    # 每个时间步的输出由两个方向的隐藏层状态组合而成: 一个处理序列从开始到结束(正向), 另一个从结束到开始(反向)
    # 定义一个方法来合并双向 RNN 的输出
    def combine_bidir(self, outs, bsz: int):
        # 将 out 重塑为一个四维张量 (num_layers, 2, batch_size, hidden_dim)
        # 2 表示双向 RNN
        # -1 表示自动计算维度大小
        # transpose(1, 2): out: (num_layers, batch_size, 2, hidden_dim)
        # contiguous(): 确保转置后的张量在内存中是连续的
        # view(): 重塑张量的形状: out: (num_layers, batch_size, 2*hidden_dim) 将正向和反向的隐藏状态拼接在一起
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def forward(self, src_tokens, **unused):
        # src_tokens: (batch_size, seq_len) 用整数表示的英语句子
        bsz, seqlen = src_tokens.size()
        # 将输入的整数表示转换为嵌入表示
        # x: (batch_size, seq_len, embed_dim)
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)
        # 将嵌入表示转换为 RNN 的输入格式: (seq_len, batch_size, input_size)
        # x: (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        # 初始化隐藏状态, h0: (num_layers * 2, batch_size, hidden_dim), 2 表示双向 RNN
        h0 = x.new_zeros(2 * self.num_layers, bsz, self.hidden_dim)
        # x 是 RNN 的输出, final_hiddens 是最后一个时间步的隐藏状态
        x, final_hiddens = self.rnn(x, h0)
        # final_hiddens: (num_layers * 2, batch_size, hidden_dim)
        # outputs: (seq_len, batch_size, 2 * hidden_dim)
        outputs = self.dropout_out_module(x)
        # 合并双向 RNN 的隐藏状态
        # 合并后的 final_hiddens: (num_layers, batch size, hidden_dim * 2)
        final_hiddens = self.combine_bidir(final_hiddens, bsz)
        # 创建编码器的填充掩码
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return tuple(
            (
                # outputs: (seq_len, batch_size, 2 * hidden_dim)
                # 每个时间步骤RNN的输出, 可以通过注意力进一步处理
                # final_hiddens: (num_layers, batch_size, hidden_dim * 2)
                # 每个时间步骤的隐藏状态, 将传递给解码器进行解码
                # encoder_padding_mask: (seq_len, batch_size)告诉解码器忽略哪些位置
                outputs,
                final_hiddens,
                encoder_padding_mask,
            )
        )

    # 排编码器输出的方法, 用于 fairseq 的 beam search, 在此不做更多探讨
    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
            )
        )


# 构建注意力机制 Attention 层
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        # input_embed_dim: 解码器的嵌入维度
        # source_embed_dim: 编码器的嵌入维度
        # output_embed_dim: 注意力层输出的维度
        # bias: 是否使用偏置
        super().__init__()
        # input_proj: 将解码器的嵌入向量投影到与编码器的嵌入向量相同的维度
        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        # output_proj: 将注意力机制的输出和原始解码器输入串联后投影到输出维度空间
        self.output_proj = nn.Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )

    def forward(self, inputs, encoder_outputs, encoder_padding_mask):
        # inputs: 来自解码器的输入 (T_seq_len, batch_size, input_embed_dim)
        # encoder_outputs: 来自编码器的输出 (S_seq_len, batch_size, source_embed_dim)
        # padding mask: 编码器输出的填充掩码 (seq_len, batch_size)
        # inputs: (T_seq_len, batch_size, input_embed_dim) => (batch_size, T_seq_len, input_embed_dim)
        # encoder_outputs: (S_seq_len, batch_size, source_embed_dim) => (batch_size, S_seq_len, source_embed_dim)
        # encoder_padding_mask: (seq_len, batch_size) => (batch_size, seq_len)
        inputs = inputs.transpose(1, 0)
        encoder_outputs = encoder_outputs.transpose(1, 0)
        encoder_padding_mask = encoder_padding_mask.transpose(1, 0)
        # 将解码器的嵌入向量投影到与编码器的嵌入向量相同的维度
        # x: (batch_size, T_seq_len, input_embed_dim) => (batch_size, T_seq_len, source_embed_dim)
        x = self.input_proj(inputs)

        # 计算注意力分数
        # bmm 是批量矩阵乘法, 计算解码器输入和编码器输出之间的点积
        # (batch_size, T_seq_len, source_embed_dim) x (batch_size, source_embed_dim, S_seq_len) = (B, T, S)
        attn_scores = torch.bmm(x, encoder_outputs.transpose(1, 2))

        # 在 encoder_padding_mask 相应位置上取消注意力分数
        if encoder_padding_mask is not None:
            # 为掩码增加一个维度, 以适应批量矩阵乘法的形状要求  (batch_size, seq_len) -> (B, 1, S)
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
            attn_scores = (
                # 在填充位置上将注意力分数设置为负无穷, 这确保了在计算 softmax 时, 填充位置不会对结果产生影响
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )

        # 对注意力分数进行 softmax 操作, 得到最终的注意力权重
        attn_scores = F.softmax(attn_scores, dim=-1)
        # 计算加权的编码器输出
        # (B, T, S) x (batch_size, S_seq_len, source_embed_dim) = (batch_size, T_seq_len, source_embed_dim)
        x = torch.bmm(attn_scores, encoder_outputs)

        # 将加权的编码器输出和原始解码器输入串联
        x = torch.cat((x, inputs), dim=-1)
        x = torch.tanh(self.output_proj(x))

        # (batch_size, T_seq_len, output_embed_dim) -> (T_seq_len, batch_size, output_embed_dim)
        return x.transpose(1, 0), attn_scores


# 构建 RNN 解码器层
class RNNDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        # 确保解码器和编码器的层数以及隐藏维度的设置是相匹配的
        assert args.decoder_layers == args.encoder_layers, f"""seq2seq rnn requires that encoder 
        and decoder have same layers of rnn. got: {args.encoder_layers, args.decoder_layers}"""
        assert args.decoder_ffn_embed_dim == args.encoder_ffn_embed_dim * 2, f"""seq2seq-rnn requires 
        that decoder hidden to be 2*encoder hidden dim. got: {args.decoder_ffn_embed_dim, args.encoder_ffn_embed_dim * 2}"""

        self.embed_dim = args.decoder_embed_dim
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers

        self.dropout_in_module = nn.Dropout(args.dropout)
        # 创建一个 GRU 网络, 设置为单向
        self.rnn = nn.GRU(
            self.embed_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=args.dropout,
            batch_first=False,
            bidirectional=False
        )
        # 创建注意力层
        self.attention = AttentionLayer(
            self.embed_dim, self.hidden_dim, self.embed_dim, bias=False
        )
        # self.attention = None
        self.dropout_out_module = nn.Dropout(args.dropout)
        # 如果隐藏层维度与嵌入维度不同, 定义一个额外的线性层以匹配嵌入维度
        if self.hidden_dim != self.embed_dim:
            self.project_out_dim = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.project_out_dim = None

        # 是否设置共享输入嵌入和输出投影的权重
        if args.share_decoder_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        # prev_output_tokens: 上一个时间步(time step)的输出
        # incremental_state: 存储和传递增量状态信息
        # 提取编码器输出、隐藏状态和填充掩码
        encoder_outputs, encoder_hiddens, encoder_padding_mask = encoder_out
        # outputs:          (seq_len, batch_size, 2 * hidden_dim)
        # encoder_hiddens:  (num_layers, batch_size, hidden_dim * 2)
        # padding_mask:     (seq_len, batch_size)

        # 根据增量状态设置初始隐藏状态
        if incremental_state is not None and len(incremental_state) > 0:
            # incremental_state 存在且非空, 即保留了上一个时间步的信息, 表示解码器正在渐进式地生成序列, 因此只需要考虑最新的输出
            # 如果保留了上一个时间步的信息，我们可以从那里继续而不是从 bos, 即从句子开头开始
            prev_output_tokens = prev_output_tokens[:, -1:]
            cache_state = self.get_incremental_state(incremental_state, "cached_state")
            prev_hiddens = cache_state["prev_hiddens"]
        else:
            # incremental_state 不存在, 则可能为模型正在训练阶段, 或者是正位于测试阶段的第一个 time step
            # 开始为输出序列做准备, 将编码器隐藏状态传至解码器隐藏状态
            prev_hiddens = encoder_hiddens

        bsz, seqlen = prev_output_tokens.size()

        # 上一步的输出转换为嵌入向量表示
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # 转换格式以适应 RNN 输入格式
        # x: B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # 存在注意力层则通过注意力层处理解码器输入和编码器输出
        if self.attention is not None:
            x, attn = self.attention(x, encoder_outputs, encoder_padding_mask)

        # 通过单向 RNN
        x, final_hiddens = self.rnn(x, prev_hiddens)
        # x = (seq_len, batch_size, hidden_dim)
        # final_hiddens =  (num_layers * directions, batch_size, hidden_dim), 这里 directions == 1
        x = self.dropout_out_module(x)

        # 如果需要，通过额外的线性层调整输出维度以匹配嵌入维度
        if self.project_out_dim != None:
            x = self.project_out_dim(x)

        # 将解码器输出映射到词汇表大小, 用于生成最终的预测
        x = self.output_projection(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # 更新增量状态以记录当前时间步的隐藏状态，这将在下一个时间步被使用
        cache_state = {
            "prev_hiddens": final_hiddens,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        return x, None

    # 用于 fairseq 的 beam search, 不展开探讨
    def reorder_incremental_state(
            self,
            incremental_state,
            new_order,
    ):
        cache_state = self.get_incremental_state(incremental_state, "cached_state")
        prev_hiddens = cache_state["prev_hiddens"]
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        cache_state = {
            "prev_hiddens": torch.stack(prev_hiddens),
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        return


# 构建 Seq2Seq 模型的整体流程
class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        # logits: 解码器的输出, 对目标词汇表的未经归一化的分数
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra


# HINT: transformer architecture
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)


# 实例化一个 Seq2Seq 模型
def build_model(args, task):
    # 从先前完成的词典创建任务中提取源语言和目标语言的词典
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # 将源语言和目标语言的 token 转换为嵌入向量
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())

    # encoder decoder
    # HINT: TODO: switch to TransformerEncoder & TransformerDecoder
    # encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # seq2seq model
    model = Seq2Seq(args, encoder, decoder)

    # initialization for seq2seq model is important, requires extra handling
    # 初始化模型的权重
    # 对于线性层(nn.Linear), 权重使用正态分布初始化, 偏置项(如果存在)设置为零
    # 对于嵌入层(nn.Embedding), 权重使用正态分布初始化, 填充索引处的权重设置为零
    # 对于多头注意力层(Multi-headAttention), 权重使用正态分布初始化
    # 对于 RNN 层(nn.RNNBase), 权重和偏置使用均匀分布初始化
    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)

    model.apply(init_params)
    return model
