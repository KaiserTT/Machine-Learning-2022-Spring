import sys
import pprint
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from seq2seq import build_model
import shutil
import sacrebleu
from LabelSmoothedCrossEntropy import LabelSmoothedCrossEntropyCriterion
from NoamOpt import NoamOpt
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from torch.nn.functional import cosine_similarity

seed = 73


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(seed)


# fairseq's beam search generator
# given model and input seqeunce, produce translation hypotheses by beam search


def main():
    # 配置参数
    config = Namespace(
        datadir="./DATA/rawdata/ted2020",
        savedir="./checkpoints/transformer",
        source_lang="en",
        target_lang="zh",
        num_workers=2,  # 获取和处理数据时使用的 CPU 线程数
        max_tokens=8192,  # 每个 batch 的最大 token 数
        accum_steps=2,  # 每 accum_steps 个 batch 更新一次参数
        lr_factor=2.,  # 用于调整最大学习率
        lr_warmup=4000,  # 学习率 warmup 的步数
        clip_norm=1.0,  # 梯度裁剪的范数上限, 用于防止梯度爆炸问题
        max_epoch=45,
        start_epoch=1,
        beam=5,  # 在进行 beam search(一种解码策略)时的 beam 大小, Beam search 是在测试阶段用于生成翻译的技术
        # 计算生成序列的最大长度, 公式为 ax + b, 其中 x 是源序列的长度
        max_len_a=1.2,
        max_len_b=10,
        post_process="sentencepiece",  # 指定解码后的文本处理方式, 这里使用 "sentencepiece" 表示移除 SentencePiece 符号
        keep_last_epochs=5,  # 保存最近几个 epoch 的模型
        resume=None,  # 如果指定了检查点文件名, 训练可以从该检查点恢复
        use_wandb=False,  # 是否使用 Weights & Biases 进行训练过程的可视化和日志记录
    )
    # 网络架构相关参数
    arch_args = Namespace(
        encoder_embed_dim=256,
        encoder_ffn_embed_dim=512,
        encoder_layers=4,
        decoder_embed_dim=256,
        decoder_ffn_embed_dim=1024,
        decoder_layers=4,
        share_decoder_input_output_embed=True,
        dropout=0.3,
    )

    # 适配 Transformer 的参数
    def add_transformer_args(args):
        args.encoder_attention_heads = 8
        args.encoder_normalize_before = True

        args.decoder_attention_heads = 8
        args.decoder_normalize_before = True

        args.activation_fn = "relu"
        args.max_source_positions = 1024
        args.max_target_positions = 1024

        # 设置 Transformer 中以上未提到的其他参数
        from fairseq.models.transformer import base_architecture
        base_architecture(arch_args)

    add_transformer_args(arch_args)

    # 设置日志: 使用 logging 模块设置日志格式和级别
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",  # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)
    if config.use_wandb:
        import wandb

        wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

    # CUDA 环境检查
    cuda_env = utils.CudaEnvironment()
    utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if config.use_wandb:
        wandb.config.update(vars(arch_args))

    # 加载数据
    # 设置翻译任务的配置
    task_cfg = TranslationConfig(
        data=config.datadir,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        train_subset="train",
        required_seq_len_multiple=8,  # 序列长度需是 8 的倍数, 处理大小为 2 的幂次方的数据时更加高效
        dataset_impl="mmap",  # 使用内存映射的方式加载数据
        upsample_primary=1,  # 主数据集的上采样倍数, 这里设置为 1, 意味着不进行上采样
    )
    task = TranslationTask.setup_task(task_cfg)
    # 记录日志信息
    logger.info("loading data for epoch 1")
    # 加载训练数据集, 如果有反向翻译数据, combine=True 表示将其与原始数据结合
    task.load_dataset(split="train", epoch=1, combine=True)
    task.load_dataset(split="valid", epoch=1)
    # 提取并打印样本数据
    sample = task.dataset("valid")[1]
    pprint.pprint(sample)
    pprint.pprint(
        "Source: " + \
        task.source_dictionary.string(
            sample['source'],
            config.post_process,
        )
    )
    pprint.pprint(
        "Target: " + \
        task.target_dictionary.string(
            sample['target'],
            config.post_process,
        )
    )

    # 从 fairseq 任务对象中加载数据迭代器
    def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
        batch_iterator = task.get_batch_iterator(
            dataset=task.dataset(split),
            max_tokens=max_tokens,
            max_sentences=None,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                max_tokens,
            ),
            ignore_invalid_inputs=True,
            seed=seed,
            num_workers=num_workers,
            epoch=epoch,
            disable_iterator_cache=not cached,
        )
        return batch_iterator

    demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
    demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
    sample = next(demo_iter)

    model = build_model(arch_args, task)
    sequence_generator = task.build_generator([model], config)
    logger.info(model)

    model = model.to(device=device)
    # generally, 0.1 is good enough
    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )
    criterion = criterion.to(device=device)

    optimizer = NoamOpt(
        model_size=arch_args.encoder_embed_dim,
        factor=config.lr_factor,
        warmup=config.lr_warmup,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")
    epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)

    def try_load_checkpoint(model, optimizer=None, name=None):
        name = name if name else "checkpoint_last.pt"
        checkpath = Path(config.savedir) / name
        if checkpath.exists():
            check = torch.load(checkpath)
            model.load_state_dict(check["model"])
            stats = check["stats"]
            step = "unknown"
            if optimizer != None:
                optimizer._step = step = check["optim"]["step"]
            logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
        else:
            logger.info(f"no checkpoints found at {checkpath}!")

    try_load_checkpoint(model, optimizer, name=config.resume)

    def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
        itr = epoch_itr.next_epoch_itr(shuffle=True)
        # 梯度累积: 每 accum_steps 个 batch 更新一次参数
        itr = iterators.GroupedIterator(itr, accum_steps)

        stats = {"loss": []}
        scaler = GradScaler()  # 使用自动混合精度(Automatic Mixed Precision, AMP)

        model.train()
        progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
        for samples in progress:
            model.zero_grad()
            accum_loss = 0
            sample_size = 0

            for i, sample in enumerate(samples):
                if i == 1:
                    # 清空 CUDA 缓存，减少内存溢出的风险
                    torch.cuda.empty_cache()

                sample = utils.move_to_cuda(sample, device=device)
                target = sample["target"]
                sample_size_i = sample["ntokens"]
                sample_size += sample_size_i

                with autocast():
                    net_output = model.forward(**sample["net_input"])
                    lprobs = F.log_softmax(net_output[0], -1)
                    loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                    accum_loss += loss.item()
                    scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            optimizer.multiply_grads(
                1 / (sample_size or 1.0))  # (sample_size or 1.0) 避免出现除以 0 的情况
            # 进行梯度裁切避免出现梯度爆炸
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # logging
            loss_print = accum_loss / sample_size
            stats["loss"].append(loss_print)
            progress.set_postfix(loss=loss_print)
            if config.use_wandb:
                wandb.log({
                    "train/loss": loss_print,
                    "train/grad_norm": gnorm.item(),
                    "train/lr": optimizer.rate(),
                    "train/sample_size": sample_size,
                })

        loss_print = np.mean(stats["loss"])
        logger.info(f"training loss: {loss_print:.4f}")
        return stats

    def decode(toks, dictionary):
        # 将张量转换为人类语言
        s = dictionary.string(
            toks.int().cpu(),
            config.post_process,
        )
        return s if s else "<unk>"

    def inference_step(sample, model):
        gen_out = sequence_generator.generate([model], sample)  # 生成机器翻译
        srcs = []  # 存储原始语言,
        hyps = []  # 存储机器翻译
        refs = []  # 存储人工翻译
        for i in range(len(gen_out)):
            srcs.append(decode(
                utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
                task.source_dictionary,
            ))
            hyps.append(decode(
                gen_out[i][0]["tokens"],  # 0 indicates using the top hypothesis in beam
                task.target_dictionary,
            ))
            refs.append(decode(
                utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
                task.target_dictionary,
            ))
        return srcs, hyps, refs

    def validate(model, task, criterion, log_to_wandb=True):
        logger.info('begin validation')
        itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

        stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
        srcs = []
        hyps = []
        refs = []

        model.eval()
        progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
        with torch.no_grad():
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample, device=device)
                net_output = model.forward(**sample["net_input"])  # 通过模型前向传播得到输出
                # 计算损失
                lprobs = F.log_softmax(net_output[0], -1)
                target = sample["target"]
                sample_size = sample["ntokens"]
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
                progress.set_postfix(valid_loss=loss.item())
                stats["loss"].append(loss)

                s, h, r = inference_step(sample, model)
                srcs.extend(s)
                hyps.extend(h)
                refs.extend(r)

        tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
        stats["loss"] = torch.stack(stats["loss"]).mean().item()
        stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)  # 计算 BLEU score
        stats["srcs"] = srcs
        stats["hyps"] = hyps
        stats["refs"] = refs

        if config.use_wandb and log_to_wandb:
            wandb.log({
                "valid/loss": stats["loss"],
                "valid/bleu": stats["bleu"].score,
            }, commit=False)

        showid = np.random.randint(len(hyps))
        logger.info("example source: " + srcs[showid])
        logger.info("example hypothesis: " + hyps[showid])
        logger.info("example reference: " + refs[showid])

        # show bleu results
        logger.info(f"validation loss:\t{stats['loss']:.4f}")
        logger.info(stats["bleu"].format())
        return stats

    def validate_and_save(model, task, criterion, optimizer, epoch, save=True):
        stats = validate(model, task, criterion)
        bleu = stats['bleu']
        loss = stats['loss']
        if save:
            # save epoch checkpoints
            savedir = Path(config.savedir).absolute()
            savedir.mkdir(parents=True, exist_ok=True)

            check = {
                "model": model.state_dict(),
                "stats": {"bleu": bleu.score, "loss": loss},
                "optim": {"step": optimizer._step}
            }
            torch.save(check, savedir / f"checkpoint{epoch}.pt")
            shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / f"checkpoint_last.pt")
            logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

            # save epoch samples
            with open(savedir / f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w",
                      encoding='utf-8') as f:
                for s, h in zip(stats["srcs"], stats["hyps"]):
                    f.write(f"{s}\t{h}\n")

            # get best valid bleu
            if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
                validate_and_save.best_bleu = bleu.score
                torch.save(check, savedir / f"checkpoint_best.pt")

            del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
            if del_file.exists():
                del_file.unlink()

        return stats

    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)

    def generate_prediction(model, task, split="test", outfile="./prediction.txt"):
        task.load_dataset(split=split, epoch=1)
        itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)

        idxs = []
        hyps = []

        model.eval()
        progress = tqdm.tqdm(itr, desc=f"prediction")
        with torch.no_grad():
            for i, sample in enumerate(progress):
                # validation loss
                sample = utils.move_to_cuda(sample, device=device)

                # do inference
                s, h, r = inference_step(sample, model)

                hyps.extend(h)
                idxs.extend(list(sample['id']))

        # sort based on the order before preprocess
        hyps = [x for _, x in sorted(zip(idxs, hyps))]

        with open(outfile, "w", encoding='utf-8') as f:
            for h in hyps:
                f.write(h + "\n")

    generate_prediction(model, task)

    import seaborn as sns

    pos_emb = model.decoder.embed_positions.weights.cpu().detach()
    print(pos_emb.size())
    ret = torch.nn.functional.cosine_similarity(pos_emb[:, None, :], pos_emb[None, :, :], dim=-1)
    print(ret.size())

    plt.figure(2, figsize=(10, 8))
    sns.heatmap(ret, cmap='viridis')
    plt.title("Positional Embedding Cosine Similarity")
    plt.savefig("ret2.png")


if __name__ == '__main__':
    main()
