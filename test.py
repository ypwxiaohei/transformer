import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sacrebleu.metrics import BLEU
import math
import numpy as np
from tqdm import tqdm
import sys

# Import model and config
from Model import Transformer
from Config import config


def main():
    # 1. 初始化设备和配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 加载tokenizer和模型
    de_tokenizer = AutoTokenizer.from_pretrained(config.de_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(config.en_tokenizer_name)

    transformer = Transformer(
        src_vocab_size=de_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout,
        device=device
    ).to(device)

    # 3. 加载训练好的模型权重
    model_path = 'models/wmt14_de-en_model1-3_epoch5.pt'  # 根据需要修改
    transformer.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    transformer.eval()
    print(f"已加载模型: {model_path}")

    # 4. 加载验证集
    validation_data = load_dataset(
        '/root/autodl-tmp/datasets/wmt14/de-en',
        split='validation'

    )

    # 5. 创建数据加载器
    val_dataloader = DataLoader(
        validation_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )

    # 6. 评估模式选择
    while True:
        print("\n请选择评估模式:")
        print("1. 计算整体BLEU和困惑度(PPL)")
        print("2. 交互式翻译")
        print("3. 退出")
        choice = input("请输入选项(1-3): ")

        if choice == '1':
            evaluate_model(transformer, val_dataloader, de_tokenizer, en_tokenizer, device)
        elif choice == '2':
            interactive_translation(transformer, de_tokenizer, en_tokenizer, device)
        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效选项，请重新输入")


def evaluate_model(model, dataloader, de_tokenizer, en_tokenizer, device):
    """计算整体BLEU和困惑度(PPL)"""
    # 1. 准备评估指标
    total_loss = 0.0
    total_tokens = 0
    all_hypotheses = []
    all_references = []
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # 2. 遍历验证集
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            # 2.1 准备batch数据
            src_texts = batch['translation']['de']
            tgt_texts = batch['translation']['en']

            src_tokens = tokenize_batch(src_texts, de_tokenizer, device)
            tgt_tokens = tokenize_batch(tgt_texts, en_tokenizer, device)

            # 2.2 模型前向传播
            outputs = model(src_tokens, tgt_tokens[:, :-1])

            # 2.3 计算损失和困惑度
            loss = criterion(
                outputs.contiguous().view(-1, outputs.shape[-1]),
                tgt_tokens[:, 1:].contiguous().view(-1)
            )
            non_pad_tokens = (tgt_tokens[:, 1:] != 0).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad_tokens

            # 2.4 生成翻译并收集结果
            for i in range(len(src_texts)):
                src = src_texts[i]
                ref = tgt_texts[i]
                hyp = greedy_decode(model, src, de_tokenizer, en_tokenizer, device)

                all_hypotheses.append(hyp)
                all_references.append([ref])

    # 3. 计算困惑度(PPL)
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"\n困惑度(PPL): {ppl:.2f}")

    # 4. 计算BLEU分数
    bleu = BLEU()
    bleu_score = bleu.corpus_score(all_hypotheses, all_references)
    print(f"BLEU分数: {bleu_score.score:.2f}")
    print(f"详细指标: {bleu_score}")


def greedy_decode(model, src_text, de_tokenizer, en_tokenizer, device, max_length=50):
    """使用贪心算法生成翻译"""
    # 1. 编码源文本
    src_tokens = de_tokenizer(
        src_text,
        truncation=True,
        max_length=config.max_seq_length,
        return_tensors='pt'
    ).input_ids.to(device)

    # 2. 初始化目标序列
    decoder_input = torch.tensor([[en_tokenizer.cls_token_id]], device=device)

    # 3. 逐步生成翻译
    for _ in range(max_length):
        with torch.no_grad():
            output = model(src_tokens, decoder_input)

        # 获取下一个token
        next_token = output[:, -1, :].argmax(dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=-1)

        # 检查结束标记
        if next_token.item() == en_tokenizer.sep_token_id:
            break

    # 4. 解码为文本
    translation = en_tokenizer.decode(
        decoder_input[0].cpu().numpy(),
        skip_special_tokens=True
    )
    return translation


def interactive_translation(model, de_tokenizer, en_tokenizer, device):
    """交互式翻译模式（添加BLEU分数计算）"""
    print("\n进入交互式翻译模式（输入'q'退出）")
    bleu = BLEU()  # 创建BLEU计算器

    while True:
        src_text = input("\n请输入德语句子: ")
        if src_text.lower() == 'q':
            break

        # 生成翻译
        translation = greedy_decode(model, src_text, de_tokenizer, en_tokenizer, device)
        print(f"英语翻译: {translation}")

        # 询问参考翻译以计算BLEU分数
        reference = input("请输入参考翻译（用于计算BLEU分数，直接回车跳过）: ")

        if reference.strip():  # 如果有参考翻译
            # 计算单句BLEU分数
            bleu_score = bleu.sentence_score(
                hypothesis=translation,
                references=[reference]  # 注意：需要将参考翻译放入列表中
            )
            print(f"BLEU分数: {bleu_score.score:.2f}")


def tokenize_batch(texts, tokenizer, device):
    """将文本批次转换为token张量"""
    tokens = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=config.max_seq_length,
        return_tensors='pt'
    ).input_ids
    return tokens.to(device)


if __name__ == "__main__":
    main()