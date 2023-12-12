import torch.nn as nn


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        # smoothing: 标签平滑的参数, 控制标签平滑的强度
        # reduce: 是否对损失进行求和
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, lprobs, target):
        # lprobs: 模型输出的对数概率 [batch_size, vocab_size]
        # target: 真实标签
        # 确保在计算交叉熵损失时，目标标签的维度与模型输出的对数概率的维度匹配
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # 计算负对数似然损失 (NLL)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        # 对模型输出的所有类别的对数概率进行求和
        # 结合标签平滑, 即不仅考虑正确类别的损失, 即nll_loss, 还要考虑其他类别的损失
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # 计算每个类别平滑的概率, 将 smoothing 值均匀分配给除了正确标签之外的所有类别
        eps_i = self.smoothing / lprobs.size(-1)
        # 结合正常的交叉熵损失和标签平滑的损失
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss
