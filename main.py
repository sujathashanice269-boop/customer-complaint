"""
主训练脚本 - 完全改进版
实现策略A：预训练用24万，训练用3.5万
✅ 修复：Label全局图预训练的维度不匹配问题
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from config import Config, get_quick_test_config, get_production_config
from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
from model import MultiModalComplaintModel, FocalLoss, ModalBalanceLoss


# ========== 源头预防: 训练监控类 ==========
class TrainingMonitor:
    """
    训练监控类 - 实时检测训练异常
    防止梯度爆炸、Loss异常、权重NaN等问题
    """

    def __init__(self, window_size=10):
        self.loss_history = []
        self.gradient_norms = []
        self.window_size = window_size
        self.nan_count = 0
        self.inf_count = 0

    def check_loss(self, loss):
        """检查loss是否异常"""
        # 检查NaN
        if torch.isnan(loss):
            self.nan_count += 1
            print(f"❌ 检测到NaN Loss! (第{self.nan_count}次)")
            return False

        # 检查Inf
        if torch.isinf(loss):
            self.inf_count += 1
            print(f"❌ 检测到Inf Loss! (第{self.inf_count}次)")
            return False

        # 检查是否突然暴涨
        if len(self.loss_history) >= self.window_size:
            recent_avg = sum(self.loss_history[-self.window_size:]) / self.window_size
            if loss.item() > recent_avg * 10:
                print(f"⚠️ Loss暴涨: {recent_avg:.4f} → {loss.item():.4f}")
                # 不停止训练，只是警告

        self.loss_history.append(loss.item())
        return True

    def check_gradients(self, model, max_norm=10.0):
        """检查梯度是否异常"""
        total_norm = 0.0
        nan_params = []
        inf_params = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                # 检查梯度是否有NaN或Inf
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                if torch.isinf(param.grad).any():
                    inf_params.append(name)

                # 计算梯度范数
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

        # 报告异常
        if nan_params:
            print(f"❌ 以下参数的梯度包含NaN: {nan_params[:3]}...")
            return False

        if inf_params:
            print(f"❌ 以下参数的梯度包含Inf: {inf_params[:3]}...")
            return False

        if total_norm > max_norm * 2:
            print(f"⚠️ 梯度范数过大: {total_norm:.2f} (阈值: {max_norm})")

        return True

    def check_model_weights(self, model):
        """检查模型权重是否异常"""
        nan_weights = []
        inf_weights = []

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_weights.append(name)
            if torch.isinf(param).any():
                inf_weights.append(name)

        if nan_weights:
            print(f"❌ 以下权重包含NaN: {nan_weights[:3]}...")
            return False

        if inf_weights:
            print(f"❌ 以下权重包含Inf: {inf_weights[:3]}...")
            return False

        return True

    def get_summary(self):
        """获取监控摘要"""
        if len(self.loss_history) == 0:
            return "无监控数据"

        recent_losses = self.loss_history[-self.window_size:]
        avg_loss = sum(recent_losses) / len(recent_losses)

        if len(self.gradient_norms) > 0:
            recent_grads = self.gradient_norms[-self.window_size:]
            avg_grad = sum(recent_grads) / len(recent_grads)
        else:
            avg_grad = 0.0

        return f"近期平均Loss: {avg_loss:.4f}, 梯度范数: {avg_grad:.2f}, NaN次数: {self.nan_count}, Inf次数: {self.inf_count}"
def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain_text_stage1(config, processor, save_dir):
    """
    Text预训练阶段1: 纯MLM领域适应
    使用24万数据 - 修复版
    """
    print("\n" + "=" * 60)
    print(f"🎯 Text预训练阶段1: 纯MLM")
    print(f"   轮数: {config.pretrain.stage1_epochs}")
    print("=" * 60)

    from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader
    import pandas as pd
    from tqdm import tqdm

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)

    # ✅ 先初始化num_added
    num_added = 0
    original_vocab_size = len(tokenizer)  # 也提前获取，避免后面未定义

    # ============================================================
    # ⭐ 关键修改1: 添加用户词典到tokenizer
    # ============================================================
    if hasattr(processor, 'user_dict_whitelist') and processor.user_dict_whitelist:
        user_words = list(processor.user_dict_whitelist)
        print(f"\n📚 添加用户词典到BERT词表:")
        print(f"  用户词数量: {len(user_words)}")

        num_added = tokenizer.add_tokens(user_words)
        print(f"  添加了 {num_added} 个新词到BERT词表")
        print(f"  词表大小: {original_vocab_size} → {len(tokenizer)}")

        # 测试分词
        test_text = "多次投诉信号差问题未解决"
        tokens = tokenizer.tokenize(test_text)
        print(f"\n  测试分词: {test_text}")
        print(f"  结果: {tokens}\n")

    # 加载BERT模型
    model = BertForMaskedLM.from_pretrained(config.model.bert_model_name)

    # ✅ 重要: 调整embedding层大小以匹配新词表
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

        # ⭐ 关键修改1: 稳定初始化新词embedding
        with torch.no_grad():
            # 获取embedding层
            embeddings = model.get_input_embeddings()

            # 计算原始词的embedding均值和标准差
            original_embeddings = embeddings.weight[:original_vocab_size]
            mean = original_embeddings.mean(dim=0)
            std = original_embeddings.std(dim=0)

            # 用更小的标准差初始化新词 (0.02 → 0.01)
            new_embeddings = embeddings.weight[original_vocab_size:]
            new_embeddings.normal_(mean=0.0, std=0.01)

            # 也可以选择用原始embedding的平均值初始化(更保守)
            # new_embeddings.copy_(mean.unsqueeze(0).expand(num_added, -1))

        print(f"  调整model embedding层: {original_vocab_size} → {len(tokenizer)}")
        print(f"  ✓ 新词embedding已用更稳定的初始化 (std=0.01)\n")

    model = model.to(config.training.device)

    # ============================================================
    # ⭐ 关键修改2: 直接读取并清洗原始文本
    # ============================================================
    print(f"📂 加载并清洗预训练数据: {config.training.large_data_file}")

    # 读取Excel
    df = pd.read_excel(config.training.large_data_file)
    print(f"  原始数据量: {len(df)}")

    # 提取文本列
    raw_texts = df['biz_cntt'].fillna('').astype(str).tolist()

    # 清洗文本
    print(f"  正在清洗文本...")
    cleaned_texts = []
    for text in tqdm(raw_texts, desc="清洗文本"):
        cleaned = processor.clean_text_smart(text)
        if cleaned:  # 只保留非空文本
            cleaned_texts.append(cleaned)

    print(f"  清洗后数据量: {len(cleaned_texts)}")
    print(f"  示例文本:")
    for i, text in enumerate(cleaned_texts[:3]):
        print(f"    {i + 1}. {text[:60]}...")
    print()

    # ============================================================
    # ⭐ 关键修改3: 传入清洗后的文本字符串
    # ============================================================
    from pretrain_tasks import TextPretrainDataset

    pretrain_dataset = TextPretrainDataset(
        texts=cleaned_texts,  # ✅ 传入清洗后的文本字符串列表
        tokenizer=tokenizer,
        max_length=config.model.bert_max_length,
        mask_prob=config.pretrain.stage1_mask_prob,
        use_span_masking=config.pretrain.use_span_masking,
        span_length=config.pretrain.span_mask_length
    )

    dataloader = DataLoader(
        pretrain_dataset,
        batch_size=config.pretrain.pretrain_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.pretrain.stage1_lr)
    total_steps = len(dataloader) * config.pretrain.stage1_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )

    # 训练
    best_loss = float('inf')
    stage1_save_dir = os.path.join(save_dir, 'stage1')
    os.makedirs(stage1_save_dir, exist_ok=True)

    for epoch in range(config.pretrain.stage1_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.pretrain.stage1_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(config.training.device)
            attention_mask = batch['attention_mask'].to(config.training.device)
            labels = batch['mlm_labels'].to(config.training.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # ⭐ 关键修改2: NaN检测和跳过
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ 警告: Batch {batch_idx} 的loss是 {loss.item()}, 跳过此batch")
                optimizer.zero_grad()
                continue

            loss.backward()

            # ⭐ 额外保护: 检查梯度是否包含NaN
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"\n⚠️ 警告: 参数 {name} 的梯度包含NaN/Inf, 跳过此batch")
                    has_nan_grad = True
                    break

            if has_nan_grad:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - MLM Loss: {avg_loss:.4f}")

        # ⭐ 修改4-改进版: 智能保存策略
        # 只有loss正常且更好时才保存
        if not torch.isnan(torch.tensor(avg_loss)) and not torch.isinf(torch.tensor(avg_loss)):
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained(stage1_save_dir)
                tokenizer.save_pretrained(stage1_save_dir)
                print(f"  ✓ 保存最佳模型 (loss={avg_loss:.4f})")
        else:
            print(f"  ⚠️ 本epoch的loss异常({avg_loss}), 跳过保存")

    print(f"\n✅ 阶段1完成 - 最佳Loss: {best_loss:.4f}")

    # ⭐ 如果所有epoch都失败,报错并终止
    if best_loss == float('inf'):
        error_msg = (
            "\n❌ 严重错误: 所有epoch的loss都是NaN/Inf!\n"
            "可能原因:\n"
            "  1. 学习率过大导致梯度爆炸\n"
            "  2. 数据中存在极端异常值\n"
            "  3. 批次大小过小导致训练不稳定\n"
            "建议:\n"
            "  1. 降低学习率 (如 1e-5)\n"
            "  2. 增加batch size (如 16 或 32)\n"
            "  3. 检查数据质量\n"
        )
        print(error_msg)
        raise RuntimeError("预训练失败: loss始终为NaN/Inf")

    print()
    return stage1_save_dir


# ============================================================
# 修改: Text预训练阶段2 - 对比学习版本
# 原来: MLM + Classification
# 现在: Supervised Contrastive Learning
# ============================================================

def pretrain_text_stage2_supcon(config, processor=None, save_dir="./pretrained_complaint_bert_improved"):
    """
    Text预训练阶段2: Supervised Contrastive Learning

    流程:
        1. 加载阶段1预训练的BERT
        2. 创建BERTForContrastiveLearning (BERT + Projection)
        3. 加载24万数据 + Repeat complaint标签
        4. 使用BalancedBatchSampler (30% pos + 70% neg)
        5. SupCon Loss训练20轮
        6. 保存BERT (丢弃projection)

    Args:
        config: 配置对象
        save_dir: 保存路径

    Returns:
        save_dir: 预训练模型保存路径
    """

    print("=" * 70)
    print("🎯 Text预训练阶段2: Supervised Contrastive Learning")
    print("=" * 70)
    print(f"方法: 监督对比学习 (SupCon)")
    print(f"轮数: {config.pretrain.stage2_epochs}")
    print(f"Batch size: {config.pretrain.pretrain_batch_size}")
    print(f"温度参数: {config.pretrain.contrastive_temperature}")
    print(f"投影维度: 128")
    print(f"正样本比例: 30%")
    print("=" * 70 + "\n")
    # ===== 0. 创建processor(如果未传入) =====
    if processor is None:
        from data_processor import ComplaintDataProcessor
        processor = ComplaintDataProcessor(config)
        print("✓ 已创建新processor")
    # ===== 1. 加载数据 =====
    print("📂 Step 1: 加载预训练数据")
    print(f"文件: {config.training.large_data_file}")

    import pandas as pd
    from tqdm import tqdm

    df = pd.read_excel(config.training.large_data_file)
    print(f"✓ 原始数据量: {len(df)}")

    # 提取文本和标签
    texts = df['biz_cntt'].fillna('').astype(str).tolist()
    labels = df['Repeat complaint'].astype(int).tolist()

    # 统计
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    print(f"✓ 数据加载完成:")
    print(f"  - 有效样本: {len(texts)}")
    print(f"  - 重复投诉: {num_pos} ({num_pos / len(labels) * 100:.2f}%)")
    print(f"  - 非重复: {num_neg} ({num_neg / len(labels) * 100:.2f}%)")
    print()

    # ===== 2. 创建模型 =====
    print("🔨 Step 2: 创建模型")

    # 2.1 确定阶段1模型路径
    stage1_dir = os.path.join(config.training.pretrain_save_dir, "stage1")

    if not os.path.exists(stage1_dir):
        print(f"⚠️  阶段1模型不存在: {stage1_dir}")
        print(f"使用原始BERT: {config.model.bert_model_name}")
        stage1_dir = config.model.bert_model_name
    else:
        print(f"✓ 加载阶段1模型: {stage1_dir}")

    # 2.2 创建对比学习模型
    from model import BERTForContrastiveLearning, SupConLoss

    model = BERTForContrastiveLearning(
        bert_model_name=stage1_dir,
        projection_dim=128
    )
    model.to(config.training.device)

    print(f"✓ 模型创建完成")
    print(f"  - 设备: {config.training.device}")
    print()

    # ===== 3. 创建数据加载器 =====
    print("📦 Step 3: 创建数据加载器")

    from transformers import BertTokenizer
    from data_processor import BalancedBatchSampler, ContrastiveTextDataset
    from torch.utils.data import DataLoader

    # 3.1 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)

    # 3.2 Dataset
    dataset = ContrastiveTextDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config.model.bert_max_length
    )

    # 3.3 BalancedBatchSampler ⭐ 核心!
    batch_sampler = BalancedBatchSampler(
        labels=labels,
        batch_size=config.pretrain.pretrain_batch_size,
        pos_ratio=0.3,  # 30%重复投诉
        shuffle=True
    )

    # 3.4 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True if config.training.device == 'cuda' else False
    )

    print(f"✓ 数据加载器创建完成")
    print(f"  - Batch数/epoch: {len(dataloader)}")
    print(f"  - 每epoch样本数: {len(dataloader) * config.pretrain.pretrain_batch_size}")
    print()

    # ===== 4. 创建优化器 =====
    print("⚙️  Step 4: 创建优化器和调度器")

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(
        model.parameters(),
        lr=config.pretrain.stage2_lr,
        weight_decay=0.01
    )

    # 学习率调度: warmup + linear decay
    total_steps = len(dataloader) * config.pretrain.stage2_epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # SupCon Loss
    criterion = SupConLoss(temperature=config.pretrain.contrastive_temperature)

    print(f"✓ 优化器创建完成")
    print(f"  - 学习率: {config.pretrain.stage2_lr}")
    print(f"  - Warmup steps: {warmup_steps}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - 温度参数: {config.pretrain.contrastive_temperature}")
    print()

    # ===== 5. 训练循环 =====
    print("🚀 Step 5: 开始训练")
    print("=" * 70 + "\n")

    best_loss = float('inf')

    for epoch in range(config.pretrain.stage2_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{config.pretrain.stage2_epochs}"
        )

        for batch in progress_bar:
            # 数据移到设备
            input_ids = batch['input_ids'].to(config.training.device)
            attention_mask = batch['attention_mask'].to(config.training.device)
            labels_batch = batch['label'].to(config.training.device)

            # 前向传播: 获取归一化的投影特征
            features = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_projection=True  # ⭐ 返回128维投影
            )
            # features: [batch, 128], 已L2归一化

            # 计算SupCon Loss
            loss = criterion(features, labels_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # 统计
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Epoch结束统计
        avg_loss = total_loss / num_batches

        print(f"\nEpoch {epoch + 1}/{config.pretrain.stage2_epochs}")
        print(f"  Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss

            # ⚠️ 关键: 只保存BERT，丢弃projection!
            bert_only = model.get_bert_only()

            # ✅ 修复1: 同时保存到父目录和stage2目录
            # 父目录保存(供课程学习加载)
            bert_only.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            # stage2子目录保存(保持兼容性)
            stage2_save_dir = os.path.join(save_dir, 'stage2')
            os.makedirs(stage2_save_dir, exist_ok=True)
            bert_only.save_pretrained(stage2_save_dir)
            tokenizer.save_pretrained(stage2_save_dir)

            print(f"  ✓ 保存最佳模型 (loss={avg_loss:.4f})")
            print(f"    父目录: {save_dir}")
            print(f"    子目录: {stage2_save_dir}")

        print()

    print("=" * 70)
    print(f"✅ 阶段2训练完成!")
    print(f"最佳Loss: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")
    print("=" * 70 + "\n")
    # ✅ 修复2: 保存processor到父目录(供课程学习加载)
    if hasattr(processor, 'save'):
        processor_save_path = os.path.join(save_dir, 'processor.pkl')
        processor.save(processor_save_path)
        print(f"✅ Processor已保存到父目录: {processor_save_path}")

    print("=" * 70 + "\n")
    return save_dir


def pretrain_label_regression(config, save_dir="./pretrained_label_regression"):
    """
    Label预训练 - 回归方案

    任务: 预测标签路径的重复投诉率

    目标:
        - 让label_encoder学习路径→风险映射
        - 输出256维向量，包含判别信息
        - 用于下游三模态融合

    数据准备:
        1. 统计每个标签路径的历史重复率
        2. 创建 (路径, 重复率) 训练对
        3. 回归训练
    """
    print("\n" + "=" * 60)
    print("🌳 Label预训练 - 路径风险回归")
    print(f"   轮数: {config.pretrain.global_graph_epochs}")
    print("=" * 60)

    device = config.training.device

    # ========== 1. 数据准备 ==========
    import pandas as pd
    from data_processor import ComplaintDataProcessor

    processor = ComplaintDataProcessor(config)

    # 加载数据并构建全局图
    df = processor.load_data(config.training.large_data_file, for_pretrain=True)
    processor.build_global_ontology_tree(df['Complaint label'].tolist())

    vocab_size = len(processor.node_to_id)
    print(f"\n📊 全局图统计:")
    print(f"   节点数: {vocab_size}")
    print(f"   边数: {processor.edge_index.shape[1]}")

    # ========== 2. 统计标签路径的重复率 ==========
    print("\n🔢 统计标签路径的历史重复率...")

    # ✅ 修复3: 改进路径统计逻辑 (处理空格问题)
    path_repeat_stats = {}

    for idx, row in df.iterrows():
        label_path = row['Complaint label']
        is_repeat = int(row['Repeat complaint'])

        # 将路径字符串转换为标准格式
        if isinstance(label_path, str) and label_path.strip():
            # ✅ 关键修复: 先移除所有空格,再统一分隔符
            # 原始: "投诉 → 网络质量 → 信号差"
            # 处理后: "投诉→网络质量→信号差"
            cleaned_path = label_path.replace(' ', '')  # 移除所有空格
            normalized_path = cleaned_path.replace('>', '→').replace('->', '→').strip()

            # 只统计包含层级关系的路径(至少有→符号)
            if '→' in normalized_path:
                if normalized_path not in path_repeat_stats:
                    path_repeat_stats[normalized_path] = {'total': 0, 'repeat': 0}

                path_repeat_stats[normalized_path]['total'] += 1
                path_repeat_stats[normalized_path]['repeat'] += is_repeat

    print(f"✅ 收集到原始路径: {len(path_repeat_stats)} 条")

    # 计算重复率,过滤样本太少的路径
    min_samples = 5
    path_risk_dict = {}

    for path, stats in path_repeat_stats.items():
        total = stats['total']
        repeat = stats['repeat']

        if total >= min_samples:
            risk = repeat / total
            path_risk_dict[path] = risk

    print(f"✅ 过滤后有效路径: {len(path_risk_dict)} 条 (样本 >= {min_samples})")

    if len(path_risk_dict) == 0:
        print("\n⚠️ 警告: 没有符合条件的路径!")
        print("   可能原因:")
        print("   1. 标签格式不匹配 - 检查是否使用→分隔")
        print("   2. 数据量太小 - 每条路径需要至少5个样本")
        print(f"   3. 数据示例: {df['Complaint label'].head(3).tolist()}")
        print("\n跳过Label预训练,继续后续流程...")
        return None  # 返回None而不是抛出错误

    print(f"\n✅ 路径统计完成:")
    print(f"   总路径数: {len(path_repeat_stats)}")
    print(f"   有效路径数: {len(path_risk_dict)} (样本 >= {min_samples})")

    # 风险分布
    risks = [v['risk'] for v in path_risk_dict.values()]
    print(f"\n📈 风险分布:")
    print(f"   均值: {np.mean(risks):.4f}")
    print(f"   中位数: {np.median(risks):.4f}")
    print(f"   标准差: {np.std(risks):.4f}")
    print(f"   最小值: {np.min(risks):.4f}")
    print(f"   最大值: {np.max(risks):.4f}")

    # ========== 3. 创建训练数据 ==========
    # 将路径转换为图数据
    train_data_list = []
    train_risks = []

    for path, risk in path_risk_dict.items():
        # Split path and clean spaces
        nodes = [n.strip() for n in path.split('→') if n.strip()]

        # Map node names to IDs
        node_ids = []
        for node_name in nodes:
            if node_name in processor.node_to_id:
                node_ids.append(processor.node_to_id[node_name])

        # Only add if path has multiple nodes
        if len(node_ids) > 1:
            train_data_list.append(node_ids)
            train_risks.append(risk)  # Use 'risk' not 'stats'

    print(f"\n✅ 训练数据准备完成: {len(train_data_list)} 条路径")
    # ✅ 添加：空数据集检查
    if len(train_data_list) == 0:
        print("❌ 错误：没有有效的训练数据！")
        print("   可能原因：")
        print("   1. 标签路径格式不匹配（缺少→符号）")
        print("   2. 有效样本数太少（< 5个样本的路径被过滤）")
        print("   3. processor.node_to_id 中没有对应的节点")
        return None, None, None

    if len(train_data_list) < 10:
        print(f"⚠️  警告：训练数据太少 ({len(train_data_list)}条)")
        print("   建议至少50条路径才能获得稳定的预训练效果")
        print("   可以尝试降低 min_samples 参数（当前为5）")
    # ========== 4. 创建模型 ==========
    from model import GATLabelEncoder, LabelRiskRegressor

    label_encoder = GATLabelEncoder(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=3,
        num_heads=4
    ).to(device)

    regressor = LabelRiskRegressor(label_encoder, hidden_dim=256).to(device)

    # ========== 5. 训练设置 ==========
    from torch.utils.data import Dataset, DataLoader

    class PathRiskDataset(Dataset):
        """路径-风险数据集"""

        def __init__(self, paths, risks, processor):
            self.paths = paths
            self.risks = risks
            self.processor = processor

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            node_ids = self.paths[idx]
            risk = self.risks[idx]

            # 构造边（顺序连接）
            edge_index = []
            if len(node_ids) > 1:
                for i in range(len(node_ids) - 1):
                    edge_index.append([node_ids[i], node_ids[i + 1]])

            # 节点层级
            node_levels = []
            for node_id in node_ids:
                # 从全局node_levels获取
                if node_id < len(self.processor.node_levels):
                    node_levels.append(self.processor.node_levels[node_id].item())
                else:
                    node_levels.append(0)

            return {
                'node_ids': torch.tensor(node_ids, dtype=torch.long),
                'edge_index': torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0),
                                                                                                            dtype=torch.long),
                'node_levels': torch.tensor(node_levels, dtype=torch.long),
                'risk': torch.tensor(risk, dtype=torch.float32)
            }

    dataset = PathRiskDataset(train_data_list, train_risks, processor)

    # 80/20划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_graph_batch)

    # ========== 6. 训练 ==========
    optimizer = torch.optim.Adam(regressor.parameters(), lr=config.pretrain.global_graph_lr)
    criterion = nn.MSELoss()  # 回归损失

    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config.pretrain.global_graph_epochs):
        # 训练阶段
        regressor.train()
        train_loss = 0
        train_mae = 0  # 平均绝对误差

        for batch in train_loader:
            batch = batch.to(device)
            # ✅ 添加：确保属性存在
            if not hasattr(batch, 'node_ids'):
                batch.node_ids = batch.x
            if not hasattr(batch, 'risk'):
                batch.risk = batch.y
            # 前向传播
            pred_risk, _ = regressor(
                batch.node_ids,
                batch.edge_index,
                batch.node_levels,
                batch.batch
            )

            # 计算损失
            loss = criterion(pred_risk.squeeze(), batch.risk)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mae += torch.abs(pred_risk.squeeze() - batch.risk).mean().item()

        train_loss /= len(train_loader)
        train_mae /= len(train_loader)

        # 验证阶段
        regressor.eval()
        val_loss = 0
        val_mae = 0
        all_preds = []
        all_risks = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # ✅ 添加：确保属性存在
                if not hasattr(batch, 'node_ids'):
                    batch.node_ids = batch.x
                if not hasattr(batch, 'risk'):
                    batch.risk = batch.y

                pred_risk, _ = regressor(
                    batch.node_ids,
                    batch.edge_index,
                    batch.node_levels,
                    batch.batch
                )

                loss = criterion(pred_risk.squeeze(), batch.risk)
                val_loss += loss.item()
                val_mae += torch.abs(pred_risk.squeeze() - batch.risk).mean().item()

                all_preds.extend(pred_risk.squeeze().cpu().numpy())
                all_risks.extend(batch.risk.cpu().numpy())

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        # 计算相关系数
        corr = np.corrcoef(all_preds, all_risks)[0, 1]

        print(f"Epoch {epoch + 1}/{config.pretrain.global_graph_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {corr:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'label_encoder': label_encoder.state_dict(),
                'regressor': regressor.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'corr': corr
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")

    # ========== 7. 阈值校准 ==========
    print("\n" + "=" * 60)
    print("📊 阈值校准")
    print("=" * 60)

    from model import ThresholdCalibrator
    calibrator = ThresholdCalibrator()

    # 将风险分数转换为二分类标签（用于校准）
    all_risks_np = np.array(all_risks)
    all_labels = (all_risks_np > 0.5).astype(int)  # 简单阈值

    optimal_threshold = calibrator.calibrate(np.array(all_preds), all_labels)

    # 保存校准器
    import pickle
    with open(os.path.join(save_dir, 'calibrator.pkl'), 'wb') as f:
        pickle.dump(calibrator, f)

    print(f"\n✅ Label预训练完成!")
    print(f"   最佳验证Loss: {best_val_loss:.4f}")
    print(f"   校准阈值: {optimal_threshold:.4f}")
    print(f"   模型保存至: {save_dir}")

    return label_encoder, regressor, calibrator


def collate_graph_batch(batch):
    """
    将多个图样本合并为一个批次
    修复版：添加必要的属性以避免AttributeError
    """
    from torch_geometric.data import Data, Batch

    data_list = []
    for item in batch:
        data = Data(
            x=item['node_ids'],
            edge_index=item['edge_index'],
            node_levels=item['node_levels'],
            y=item['risk']
        )
        data_list.append(data)

    batched = Batch.from_data_list(data_list)

    # ✅ 修复：添加必要的属性
    batched.node_ids = batched.x  # 将x复制为node_ids
    batched.risk = batched.y  # 将y复制为risk

    return batched


def train_curriculum_learning(config, processor, train_data, val_data, pretrain_text_path=None):
    """
    课程学习训练 - 方向五
    使用3.5万平衡数据
    """
    print("\n" + "=" * 60)
    print("🚀 课程学习训练")
    print("=" * 60)

    # ========== 单模态快速测试模式 ==========
    import sys
    test_single_modal = None
    for arg in sys.argv:
        if '--test_single_modal' in arg:
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                test_single_modal = sys.argv[idx + 1]
                break

    if test_single_modal:
        print(f"\n⚡ 快速测试模式: 只训练 {test_single_modal}_only")
        print("=" * 60)

    vocab_size = train_data['vocab_size']
    device = config.training.device

    # 创建数据集
    train_dataset = ComplaintDataset(
        text_data=train_data['text_data'],
        node_ids_list=train_data['node_ids_list'],
        edges_list=train_data['edges_list'],
        node_levels_list=train_data['node_levels_list'],
        struct_features=train_data['struct_features'],
        targets=train_data['targets']
    )

    val_dataset = ComplaintDataset(
        text_data=val_data['text_data'],
        node_ids_list=val_data['node_ids_list'],
        edges_list=val_data['edges_list'],
        node_levels_list=val_data['node_levels_list'],
        struct_features=val_data['struct_features'],
        targets=val_data['targets']
    )

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                              shuffle=True, num_workers=0, collate_fn=custom_collate_fn,drop_last=True)  # ✅ 新增
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size,
                            shuffle=False, num_workers=0, collate_fn=custom_collate_fn)  # ✅ 新增

    # ========== 阶段1: 单模态预训练 ==========
    print("\n📌 阶段1: 单模态预训练")

    models = {}
    for mode in ['text_only', 'label_only', 'struct_only']:
        print(f"\n训练 {mode} 模型...")

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode=mode,
            pretrained_path=pretrain_text_path
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.training.stage1_lr)
        criterion = FocalLoss() if config.training.use_focal_loss else nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(config.training.stage1_single_modal_epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                # ✅ 修复：更严格的模态判断
                need_text = mode in ['text_only', 'text_label', 'text_struct', 'full']
                input_ids = batch['input_ids'].to(device) if need_text else None
                attention_mask = batch['attention_mask'].to(device) if need_text else None

                need_label = mode in ['label_only', 'text_label', 'label_struct', 'full']
                node_ids_list = batch['node_ids'] if need_label else None
                edges_list = batch['edges'] if need_label else None
                node_levels_list = batch['node_levels'] if need_label else None

                need_struct = mode in ['struct_only', 'text_struct', 'label_struct', 'full']
                struct_features = batch['struct_features'].to(device) if need_struct else None

                targets = batch['target'].to(device)

                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=node_ids_list,
                    edges_list=edges_list,
                    node_levels_list=node_levels_list,
                    struct_features=struct_features
                )

                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()

            # 验证
            val_acc = evaluate_model(model, val_loader, device, mode)
            print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                models[mode] = model.state_dict()

        print(f"✅ {mode} 最佳准确率: {best_acc:.4f}")
        # 如果是单模态测试，训练完就返回
        if test_single_modal and mode == f"{test_single_modal}_only":
            print(f"\n✅ 单模态测试完成: {mode}")
            print(f"   最佳准确率: {best_acc:.4f}")
            print("\n💡 如果没有报错，说明这个模态的代码没问题！")
            return None, None

    # ========== 阶段2: 双模态交互 ==========
    print("\n📌 阶段2: 双模态交互")

    dual_modes = ['text_label', 'text_struct', 'label_struct']
    for mode in dual_modes:
        print(f"\n训练 {mode} 模型...")

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode=mode,
            pretrained_path=pretrain_text_path
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.training.stage2_lr)
        criterion = FocalLoss() if config.training.use_focal_loss else nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(config.training.stage2_dual_modal_epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                # ✅ 修复：更严格的模态判断
                need_text = mode in ['text_only', 'text_label', 'text_struct', 'full']
                input_ids = batch['input_ids'].to(device) if need_text else None
                attention_mask = batch['attention_mask'].to(device) if need_text else None

                need_label = mode in ['label_only', 'text_label', 'label_struct', 'full']
                node_ids_list = batch['node_ids'] if need_label else None
                edges_list = batch['edges'] if need_label else None
                node_levels_list = batch['node_levels'] if need_label else None

                need_struct = mode in ['struct_only', 'text_struct', 'label_struct', 'full']
                struct_features = batch['struct_features'].to(device) if need_struct else None

                targets = batch['target'].to(device)

                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=node_ids_list,
                    edges_list=edges_list,
                    node_levels_list=node_levels_list,
                    struct_features=struct_features
                )

                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()

            # 验证
            val_acc = evaluate_model(model, val_loader, device, mode)
            print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                models[mode] = model.state_dict()

        print(f"✅ {mode} 最佳准确率: {best_acc:.4f}")

    # ========== 阶段3: 三模态融合 ==========
    print("\n📌 阶段3: 三模态融合（Full模型）")

    full_model = MultiModalComplaintModel(
        config=config,
        vocab_size=vocab_size,
        mode='full',
        pretrained_path=pretrain_text_path
    ).to(device)

    optimizer = optim.AdamW(full_model.parameters(), lr=config.training.stage3_lr)
    criterion = FocalLoss() if config.training.use_focal_loss else nn.CrossEntropyLoss()

    # 模态平衡损失（新版 - 基于注意力权重）
    modal_balance_loss_fn = ModalBalanceLoss(weight=config.training.modal_balance_weight) \
        if config.training.use_modal_balance_loss else None
    # 模态平衡损失
    modal_balance_loss_fn = ModalBalanceLoss(weight=config.training.modal_balance_weight) \
        if config.training.use_modal_balance_loss else None

    # ========== 源头预防: 启动前验证模型返回格式 ==========
    print("\n🔍 验证模型返回格式...")
    try:
        # 获取一个测试batch
        test_batch = next(iter(train_loader))
        test_size = min(2, len(test_batch['input_ids']))

        # 测试1: return_attention=False
        with torch.no_grad():
            result1 = full_model(
                input_ids=test_batch['input_ids'][:test_size].to(device),
                attention_mask=test_batch['attention_mask'][:test_size].to(device),
                node_ids_list=test_batch['node_ids'][:test_size],
                edges_list=test_batch['edges'][:test_size],
                node_levels_list=test_batch['node_levels'][:test_size],
                struct_features=test_batch['struct_features'][:test_size].to(device),
                return_attention=False
            )

        # 检查返回格式
        if isinstance(result1, tuple) and len(result1) == 2:
            print("  ✅ return_attention=False 返回格式正确: (logits, None)")
        else:
            print(f"  ⚠️ 返回格式: {type(result1)}")

        # 测试2: return_attention=True
        with torch.no_grad():
            result2 = full_model(
                input_ids=test_batch['input_ids'][:test_size].to(device),
                attention_mask=test_batch['attention_mask'][:test_size].to(device),
                node_ids_list=test_batch['node_ids'][:test_size],
                edges_list=test_batch['edges'][:test_size],
                node_levels_list=test_batch['node_levels'][:test_size],
                struct_features=test_batch['struct_features'][:test_size].to(device),
                return_attention=True
            )

        if isinstance(result2, tuple) and len(result2) == 2:
            logits, attn = result2
            if isinstance(attn, dict):
                required_keys = ['text_to_label', 'label_to_text', 'semantic_to_struct', 'struct_to_semantic']
                missing = [k for k in required_keys if k not in attn or attn[k] is None]
                if missing:
                    print(f"  ⚠️ 注意力权重缺少keys: {missing}")
                else:
                    print("  ✅ return_attention=True 返回格式正确，所有keys完整")

                    # 测试模态平衡损失（现在modal_balance_loss_fn已经定义了！）
                    if modal_balance_loss_fn is not None:
                        try:
                            test_balance = modal_balance_loss_fn(attn)
                            print(f"  ✅ 模态平衡损失计算正常: {test_balance.item():.4f}")
                        except Exception as e:
                            print(f"  ⚠️ 模态平衡损失测试失败: {e}")
                    else:
                        print("  ℹ️ 模态平衡损失未启用，跳过测试")
            else:
                print(f"  ⚠️ attention_weights类型错误: {type(attn)}")

        print("✅ 模型验证通过，开始训练...\n")

    except Exception as e:
        print(f"⚠️ 模型验证出现问题: {e}")
        print("继续训练，但请注意观察...")

    # ========== 初始化训练监控 ==========
    monitor = TrainingMonitor(window_size=10)
    print("✅ 训练监控已启动")

    best_acc = 0
    best_model_path = os.path.join(config.training.save_dir, 'best_full_model.pth')
    print(f"\n🎯 模态平衡策略:")
    if modal_balance_loss_fn:
        print("  ✅ 启用注意力权重平衡（熵最大化）")
        print("  📊 目标：text=33%, label=33%, struct=33%")
    else:
        print("  ⚠️  未启用模态平衡")

    best_acc = 0
    best_model_path = os.path.join(config.training.save_dir, 'best_full_model.pth')
    # ========== 源头预防: 初始化训练监控 ==========
    monitor = TrainingMonitor(window_size=10)
    print("✅ 训练监控已启动")
    for epoch in range(config.training.stage3_full_epochs):
        full_model.train()
        total_loss = 0
        total_cls_loss = 0
        total_balance_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            # ✅ Full模式需要所有特征
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_ids_list = batch['node_ids']
            edges_list = batch['edges']
            node_levels_list = batch['node_levels']
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].to(device)

            # ========== 前向传播（返回注意力权重）==========
            # 如果启用模态平衡，需要返回attention_weights
            if modal_balance_loss_fn is not None:
                logits, attention_weights = full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=node_ids_list,
                    edges_list=edges_list,
                    node_levels_list=node_levels_list,
                    struct_features=struct_features,
                    return_attention=True  # ← 关键：返回注意力权重
                )
            else:
                # 不需要attention_weights，加快速度
                logits, _ = full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=node_ids_list,
                    edges_list=edges_list,
                    node_levels_list=node_levels_list,
                    struct_features=struct_features
                )
                attention_weights = None

            # ========== 分类损失 ==========
            cls_loss = criterion(logits, targets)

            # ========== 模态平衡损失（新版）==========
            if modal_balance_loss_fn is not None and attention_weights is not None:
                try:
                    # 使用新的基于注意力权重的方法
                    balance_loss = modal_balance_loss_fn(attention_weights)

                    # 总损失
                    loss = cls_loss + balance_loss
                    total_balance_loss += balance_loss.item()

                except Exception as e:
                    # 如果计算失败，只用分类损失
                    print(f"  ⚠️ 模态平衡损失计算失败: {e}")
                    loss = cls_loss
            else:
                loss = cls_loss

            # ========== 反向传播 + 监控检查 ==========
            optimizer.zero_grad()

            # 源头预防: 检查loss是否异常
            if not monitor.check_loss(loss):
                print("❌ Loss异常，跳过此batch")
                continue

            loss.backward()

            # 源头预防: 检查梯度是否异常
            if not monitor.check_gradients(full_model, max_norm=config.training.max_grad_norm):
                print("❌ 梯度异常，跳过此batch")
                optimizer.zero_grad()  # 清空异常梯度
                continue

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), config.training.max_grad_norm)

            # 源头预防: 更新前最后检查
            if not monitor.check_model_weights(full_model):
                print("❌ 模型权重异常，停止训练")
                break

            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        # 打印监控摘要
        print(f"  📊 监控: {monitor.get_summary()}")
        # 验证
        val_acc, val_metrics = evaluate_model(full_model, val_loader, device, 'full', return_metrics=True)

        print(f"Epoch {epoch + 1}/{config.training.stage3_full_epochs}")
        print(f"  Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}", end="")
        if modal_balance_loss_fn:
            print(f", Balance: {total_balance_loss / len(train_loader):.4f})", end="")
        else:
            print(")", end="")
        print(f"\n  Val Acc: {val_acc:.4f}, Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': full_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config
            }, best_model_path)
            print(f"  ✓ 保存最佳模型 (acc={val_acc:.4f})")

    print(f"\n✅ 课程学习训练完成 - 最佳准确率: {best_acc:.4f}")
    print(f"✅ 最佳模型保存在: {best_model_path}")

    return full_model, best_model_path


def evaluate_model(model, dataloader, device, mode, return_metrics=False):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # ✅ 修复：更严格的模态判断
            need_text = mode in ['text_only', 'text_label', 'text_struct', 'full']
            input_ids = batch['input_ids'].to(device) if need_text else None
            attention_mask = batch['attention_mask'].to(device) if need_text else None

            need_label = mode in ['label_only', 'text_label', 'label_struct', 'full']
            node_ids_list = batch['node_ids'] if need_label else None
            edges_list = batch['edges'] if need_label else None
            node_levels_list = batch['node_levels'] if need_label else None

            need_struct = mode in ['struct_only', 'text_struct', 'label_struct', 'full']
            struct_features = batch['struct_features'].to(device) if need_struct else None

            targets = batch['target'].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=node_ids_list,
                edges_list=edges_list,
                node_levels_list=node_levels_list,
                struct_features=struct_features
            )

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)

    if return_metrics:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary'
        )
        return accuracy, {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return accuracy


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='客户重复投诉预测系统')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'pretrain_only'],
                        help='运行模式')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式')
    parser.add_argument('--production', action='store_true',
                        help='生产环境模式')
    parser.add_argument('--skip_text_pretrain', action='store_true',
                        help='跳过Text预训练')
    parser.add_argument('--skip_label_pretrain', action='store_true',
                        help='跳过Label预训练')
    parser.add_argument('--test_single_modal', type=str, default=None,
                        choices=['text', 'label', 'struct'],
                        help='单独测试某个模态（用于快速调试）')  # ← 新添加这行
    parser.add_argument('--data_file', type=str, default=None,
                        help='训练数据文件')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    # 加载配置
    if args.quick_test:
        config = get_quick_test_config()
    elif args.production:
        config = get_production_config()
    else:
        config = Config()

    # 覆盖数据文件
    if args.data_file:
        config.training.data_file = args.data_file

    # 打印配置摘要
    config.print_summary()

    # 使用设备
    device = config.training.device
    print(f"使用设备: {device}\n")

    # ========== 数据准备 ==========
    print("=" * 60)
    print("📂 加载数据")
    print("=" * 60)

    processor = ComplaintDataProcessor(
        config=config,
        user_dict_file=config.data.user_dict_file
    )

    # ✅ 第一步:使用24万数据构建全局本体树
    print("\n🌳 使用24万数据构建全局本体树...")
    large_df = processor.load_data(config.training.large_data_file, for_pretrain=True)
    processor.build_global_ontology_tree(large_df['Complaint label'].tolist())

    # ✅ 关键修改1: 保存全局词汇表(在预训练开始前)
    vocab_save_path = os.path.join(config.training.pretrain_save_dir, 'stage2', 'global_vocab.pkl')
    os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
    processor.save_global_vocab(vocab_save_path)

    # 同时保存到Label预训练目录
    label_vocab_path = os.path.join(config.training.label_pretrain_save_dir, 'global_vocab.pkl')
    os.makedirs(os.path.dirname(label_vocab_path), exist_ok=True)
    processor.save_global_vocab(label_vocab_path)

    # 保存处理器(包含全局本体树)
    processor_save_path = os.path.join(config.training.pretrain_save_dir, 'processor.pkl')
    processor.save(processor_save_path)

    # ========== 预训练阶段 ==========
    if args.mode == 'pretrain_only' or not args.skip_text_pretrain:
        print("\n" + "=" * 60)
        print("🚀 预训练阶段")
        print("=" * 60)

        # Text预训练阶段1
        if not args.skip_text_pretrain:
            stage1_dir = pretrain_text_stage1(
                config, processor, config.training.pretrain_save_dir
            )

            # Text预训练阶段2
            # ✅ 修复5: 传入processor
            stage2_dir = pretrain_text_stage2_supcon(
                config,
                processor=processor,  # 传入已创建的processor
                save_dir=config.training.pretrain_save_dir
            )

            pretrain_text_path = stage2_dir
        else:
            pretrain_text_path = config.training.pretrain_save_dir

        # Label全局图预训练
            # Label全局图预训练
            if not args.skip_label_pretrain:
                print("\n" + "=" * 60)
                print("Label Pretrain - Path Risk Regression")
                print(f"Epochs: {config.pretrain.global_graph_epochs}")
                print("=" * 60)

                save_dir = os.path.join(config.training.pretrain_save_dir, "label_regression")
                label_pretrain_result = pretrain_label_regression(config, save_dir)

                # Handle None return value
                if label_pretrain_result is None:
                    print("WARNING: Label pretrain skipped, using random initialization")
                else:
                    label_encoder, regressor, calibrator = label_pretrain_result

                    if label_encoder:
                        encoder_save_path = os.path.join(config.training.label_pretrain_save_dir, "label_encoder.pt")
                        torch.save(label_encoder.state_dict(), encoder_save_path)
                        print(f"Label pretrain completed: {save_dir}")

    else:
        pretrain_text_path = config.training.pretrain_save_dir

        # ✅ 关键修改2: 如果只是预训练,确保词汇表已保存
        if args.mode == 'pretrain_only':
            print("\n" + "=" * 60)
            print("✅ 预训练完成!")
            print("=" * 60)
            print(f"📁 保存位置:")
            print(f"  - Text预训练: {pretrain_text_path}")
            print(f"  - Label预训练: {config.training.label_pretrain_save_dir}")
            print(f"  - 全局词汇表: {vocab_save_path}")
            print(f"               {label_vocab_path}")

            # 验证词汇表是否存在
            if os.path.exists(vocab_save_path):
                import pickle
                with open(vocab_save_path, 'rb') as f:
                    vocab_data = pickle.load(f)
                print(f"\n✅ 词汇表验证:")
                print(f"  - 节点数: {len(vocab_data['node_to_id'])}")
                print(f"  - 边数: {len(vocab_data['global_edges'])}")

            return

    # ========== 训练阶段 ==========
    print("\n" + "=" * 60)
    print("📊 准备训练数据（3.5万平衡数据）")
    print("=" * 60)
    # ✅ 可选: 检查是否已加载全局词汇表
    if not processor.node_to_id:
        print("⚠️  警告: 全局词汇表未加载!")
        print("   这可能是因为跳过了预训练阶段")
        print("   将使用当前数据重新构建词汇表（可能导致维度不匹配）")

        # 询问用户是否继续
        user_input = input("\n是否继续? (y/n): ")
        if user_input.lower() != 'y':
            print("已退出。请先运行预训练: python main.py --mode pretrain_only")
            return
    # 加载3.5万平衡数据
    train_val_data = processor.prepare_datasets(
        train_file=config.training.data_file,
        for_pretrain=False
    )

    # 划分训练集和验证集
    total_size = len(train_val_data['targets'])
    train_size = int(total_size * (1 - config.training.val_size))
    val_size = total_size - train_size

    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 分割数据
    def split_data(data, indices):
        return {
            'text_data': {
                'input_ids': data['text_data']['input_ids'][indices],
                'attention_mask': data['text_data']['attention_mask'][indices]
            },
            'node_ids_list': [data['node_ids_list'][i] for i in indices],
            'edges_list': [data['edges_list'][i] for i in indices],
            'node_levels_list': [data['node_levels_list'][i] for i in indices],
            'struct_features': data['struct_features'][indices],
            'targets': data['targets'][indices],
            'vocab_size': data['vocab_size']
        }

    train_data = split_data(train_val_data, train_indices)
    val_data = split_data(train_val_data, val_indices)

    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(val_indices)}")

    # 课程学习训练
    if config.training.use_curriculum_learning:
        final_model, model_path = train_curriculum_learning(
            config, processor, train_data, val_data, pretrain_text_path
        )
    else:
        print("⚠️ 暂不支持非课程学习模式，请启用课程学习")
        return

    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    print(f"最佳模型保存在: {model_path}")


if __name__ == "__main__":
    main()