"""
============================================================
补充实验脚本 - 完全符合参考文献格式（最终修复版）
============================================================

参考文献对照：
- AAFHA Fig.3: Learning Rate Sensitivity (简洁折线图，只有Accuracy)
- AAFHA Fig.4: Dropout Sensitivity (简洁折线图)
- AAFHA Fig.5: Time Complexity (表格)
- AAFHA Fig.7: ROC Curves
- AAFHA Fig.9: Confusion Matrix
- AAFHA Fig.10-11: Integration Analysis (LIME特征权重)
- AAFHA Table 5: Ablation Study (表格)
- 假新闻论文 Table 16/17: Fusion Comparison (表格)
- 假新闻论文 Table 11: Cosine Similarity (表格)

使用方法：
    python run_supplementary_experiments.py --exp all
    python run_supplementary_experiments.py --exp lr_sensitivity
    ...

修复内容：
- 维度匹配：text_feat通过text_proj从768->256，与struct_feat (256)匹配
- 空列表检查：semantic_alignment中添加空列表保护
- 特征名称：确保53个特征名称
"""

import os
import sys
import json
import time
import gc
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score
)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# 中文字体设置
# =============================================================================
def setup_font():
    """设置字体，支持中文显示"""
    font_list = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False

setup_font()

# =============================================================================
# 导入项目模块
# =============================================================================
try:
    from config import Config
    from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
    from model import MultiModalComplaintModel, FocalLoss
    print("✅ 项目模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保脚本放在项目根目录（与config.py同级）")
    sys.exit(1)


# =============================================================================
# 工具函数
# =============================================================================
def set_seed(seed=42):
    """设置随机种子，确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)


def safe_mean(lst):
    """安全计算均值，处理空列表"""
    if not lst:
        return 0.0
    return float(np.mean(lst))


def safe_std(lst):
    """安全计算标准差，处理空列表"""
    if not lst or len(lst) < 2:
        return 0.0
    return float(np.std(lst))


def safe_min(lst):
    """安全计算最小值，处理空列表"""
    if not lst:
        return 0.0
    return float(np.min(lst))


def safe_max(lst):
    """安全计算最大值，处理空列表"""
    if not lst:
        return 0.0
    return float(np.max(lst))


# =============================================================================
# 数据准备
# =============================================================================
def prepare_data(config, pretrained_path=None):
    """
    准备数据集

    Args:
        config: 配置对象
        pretrained_path: 预训练模型路径

    Returns:
        train_loader, val_loader, test_loader, vocab_size, processor
    """
    print("\n📊 准备数据...")

    # 初始化处理器
    processor = ComplaintDataProcessor(
        config=config,
        user_dict_file=config.data.user_dict_file
    )

    # 尝试加载处理器状态
    processor_paths = [
        './processor.pkl',
        './pretrained_complaint_bert_improved/processor.pkl',
        './pretrained_complaint_bert_improved/stage2/processor.pkl'
    ]
    if pretrained_path:
        processor_paths.insert(0, os.path.join(os.path.dirname(pretrained_path), 'processor.pkl'))

    for path in processor_paths:
        if os.path.exists(path):
            try:
                processor.load(path)
                print(f"✅ 加载处理器: {path}")
                break
            except Exception as e:
                print(f"⚠️ 加载处理器失败 {path}: {e}")

    # 准备数据
    data = processor.prepare_datasets(
        train_file=config.training.data_file,
        for_pretrain=False
    )

    vocab_size = data.get('vocab_size', len(processor.node_to_id) + 1)

    # 划分数据 (60% 训练, 20% 验证, 20% 测试)
    total_size = len(data['targets'])
    indices = torch.randperm(total_size).tolist()

    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    def split_data(data_dict, idx_list):
        """根据索引划分数据"""
        return {
            'text_data': {
                'input_ids': data_dict['text_data']['input_ids'][idx_list],
                'attention_mask': data_dict['text_data']['attention_mask'][idx_list]
            },
            'node_ids_list': [data_dict['node_ids_list'][i] for i in idx_list],
            'edges_list': [data_dict['edges_list'][i] for i in idx_list],
            'node_levels_list': [data_dict['node_levels_list'][i] for i in idx_list],
            'struct_features': data_dict['struct_features'][idx_list],
            'targets': data_dict['targets'][idx_list]
        }

    train_data = split_data(data, train_indices)
    val_data = split_data(data, val_indices)
    test_data = split_data(data, test_indices)

    # 创建Dataset
    train_dataset = ComplaintDataset(
        train_data['text_data'], train_data['node_ids_list'],
        train_data['edges_list'], train_data['node_levels_list'],
        train_data['struct_features'], train_data['targets']
    )
    val_dataset = ComplaintDataset(
        val_data['text_data'], val_data['node_ids_list'],
        val_data['edges_list'], val_data['node_levels_list'],
        val_data['struct_features'], val_data['targets']
    )
    test_dataset = ComplaintDataset(
        test_data['text_data'], test_data['node_ids_list'],
        test_data['edges_list'], test_data['node_levels_list'],
        test_data['struct_features'], test_data['targets']
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        collate_fn=custom_collate_fn
    )

    print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, vocab_size, processor


def quick_train_and_evaluate(model, train_loader, val_loader, test_loader, config, num_epochs=10):
    """
    快速训练并评估模型

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        config: 配置
        num_epochs: 训练轮数

    Returns:
        metrics, all_preds, all_probs, all_targets
    """
    device = config.training.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    if config.training.use_focal_loss:
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=batch['node_ids'],
                edges_list=batch['edges'],
                node_levels_list=batch['node_levels'],
                struct_features=struct_features
            )

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()

    # 测试
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=batch['node_ids'],
                edges_list=batch['edges'],
                node_levels_list=batch['node_levels'],
                struct_features=struct_features
            )

            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(batch['target'].numpy())

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'auc': roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
    }

    return metrics, all_preds, all_probs, all_targets


# =============================================================================
# 实验1: 学习率敏感性 (参考 AAFHA Fig.3)
# =============================================================================
def run_lr_sensitivity(config, pretrained_path, save_dir):
    """
    学习率敏感性分析
    参考AAFHA Fig.3格式：简洁折线图，只展示Accuracy
    """
    print("\n" + "=" * 60)
    print("实验: Learning Rate Sensitivity (参考AAFHA Fig.3)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
    results = {}

    # 保存原始学习率
    original_lr = config.training.learning_rate

    for lr in learning_rates:
        print(f"\n>>> Learning Rate: {lr}")
        config.training.learning_rate = lr

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode='full',
            pretrained_path=pretrained_path
        )

        metrics, _, _, _ = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        results[lr] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 恢复原始学习率
    config.training.learning_rate = original_lr

    # ========== 绘图 (AAFHA Fig.3 风格) ==========
    fig, ax = plt.subplots(figsize=(8, 5))

    lrs = list(results.keys())
    accs = [results[lr]['accuracy'] for lr in lrs]

    ax.plot(range(len(lrs)), accs, 'b-o', linewidth=2, markersize=8, label='Our Dataset')

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Impact of Learning Rate on Model Accuracy', fontsize=14)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in lrs])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 设置Y轴范围
    y_min = min(accs) - 0.05
    y_max = max(accs) + 0.02
    ax.set_ylim(max(0, y_min), min(1, y_max))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 保存: lr_sensitivity.png")

    return results


# =============================================================================
# 实验2: Dropout敏感性 (参考 AAFHA Fig.4)
# =============================================================================
def run_dropout_sensitivity(config, pretrained_path, save_dir):
    """
    Dropout敏感性分析
    参考AAFHA Fig.4格式：简洁折线图
    """
    print("\n" + "=" * 60)
    print("实验: Dropout Sensitivity (参考AAFHA Fig.4)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}

    # 保存原始dropout
    original_dropout = config.model.dropout

    for dropout in dropout_rates:
        print(f"\n>>> Dropout Rate: {dropout}")
        config.model.dropout = dropout

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode='full',
            pretrained_path=pretrained_path
        )

        metrics, _, _, _ = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        results[dropout] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 恢复原始dropout
    config.model.dropout = original_dropout

    # ========== 绘图 (AAFHA Fig.4 风格) ==========
    fig, ax = plt.subplots(figsize=(8, 5))

    dropouts = list(results.keys())
    accs = [results[d]['accuracy'] for d in dropouts]

    ax.plot(dropouts, accs, 'b-o', linewidth=2, markersize=8, label='Our Dataset')

    ax.set_xlabel('Dropout Rate', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Impact of Dropout Rates on Accuracy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    y_min = min(accs) - 0.05
    y_max = max(accs) + 0.02
    ax.set_ylim(max(0, y_min), min(1, y_max))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dropout_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 保存: dropout_sensitivity.png")

    return results


# =============================================================================
# 实验3: 融合方式比较 (参考假新闻论文 Table 16/17)
# =============================================================================
def run_fusion_comparison(config, pretrained_path, save_dir):
    """
    融合方式比较
    参考假新闻论文Table 16/17格式：**表格形式**
    """
    print("\n" + "=" * 60)
    print("实验: Fusion Method Comparison (参考Table 16/17)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    # 测试不同融合方式
    fusion_methods = {
        'Text+Label': 'text_label',
        'Text+Struct': 'text_struct',
        'Label+Struct': 'label_struct',
        'Full Model (Cross-Attention)': 'full'
    }

    results = {}

    for name, mode in fusion_methods.items():
        print(f"\n>>> Testing: {name}")

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode=mode,
            pretrained_path=pretrained_path
        )

        metrics, _, _, _ = quick_train_and_evaluate(
            model, train_loader, val_loader, test_loader, config, num_epochs=10
        )

        results[name] = metrics
        print(f"  Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========== 生成表格 (Table 16/17 风格) ==========
    table_data = []
    for method, metrics in results.items():
        table_data.append({
            'Model': method,
            'Accuracy': f"{metrics['accuracy'] * 100:.2f}",
            'Precision': f"{metrics['precision'] * 100:.2f}",
            'Recall': f"{metrics['recall'] * 100:.2f}",
            'F1 Score': f"{metrics['f1'] * 100:.2f}",
            'AUC': f"{metrics['auc'] * 100:.2f}"
        })

    df = pd.DataFrame(table_data)

    # 保存为CSV
    df.to_csv(os.path.join(save_dir, 'fusion_comparison_table.csv'), index=False)

    # 生成LaTeX表格
    latex_table = df.to_latex(
        index=False,
        caption='Comparative study on different fusion models',
        label='tab:fusion_comparison'
    )
    with open(os.path.join(save_dir, 'fusion_comparison_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✅ 保存: fusion_comparison_table.csv, fusion_comparison_table.tex")
    print("\n📋 融合方式比较表格:")
    print(df.to_string(index=False))

    return results


# =============================================================================
# 实验4: 时间复杂度分析 (参考 AAFHA 4.8节)
# =============================================================================
def run_time_complexity(config, pretrained_path, save_dir):
    """
    时间复杂度分析
    参考AAFHA 4.8节格式：**表格形式**
    """
    print("\n" + "=" * 60)
    print("实验: Time Complexity Analysis (参考AAFHA 4.8节)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    modes = {
        'Text Only': 'text_only',
        'Label Only': 'label_only',
        'Struct Only': 'struct_only',
        'Text+Label': 'text_label',
        'Text+Struct': 'text_struct',
        'Full Model': 'full'
    }

    results = {}

    for name, mode in modes.items():
        print(f"\n>>> Testing: {name}")

        model = MultiModalComplaintModel(
            config=config,
            vocab_size=vocab_size,
            mode=mode,
            pretrained_path=pretrained_path
        )
        model = model.to(device)
        model.eval()

        # 参数量
        num_params = count_parameters(model)

        # 推理时间测量
        inference_times = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                struct_features = batch['struct_features'].to(device)

                # 预热
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                    struct_features=struct_features
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # 计时
                start_time = time.time()
                for _ in range(5):
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        node_ids_list=batch['node_ids'],
                        edges_list=batch['edges'],
                        node_levels_list=batch['node_levels'],
                        struct_features=struct_features
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.time()
                batch_size = input_ids.shape[0]
                avg_time = (end_time - start_time) / 5 / batch_size * 1000  # ms per sample
                inference_times.append(avg_time)

                if batch_idx >= 2:  # 只测3个batch
                    break

        results[name] = {
            'parameters': num_params,
            'parameters_M': num_params / 1e6,
            'inference_time_ms': safe_mean(inference_times)
        }

        print(f"  Parameters: {num_params / 1e6:.2f}M, Inference: {safe_mean(inference_times):.2f}ms")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========== 生成表格 ==========
    table_data = []
    for name, data in results.items():
        table_data.append({
            'Model': name,
            'Parameters (M)': f"{data['parameters_M']:.2f}",
            'Inference Time (ms)': f"{data['inference_time_ms']:.2f}"
        })

    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'time_complexity_table.csv'), index=False)

    # LaTeX
    latex_table = df.to_latex(
        index=False,
        caption='Time complexity analysis',
        label='tab:time_complexity'
    )
    with open(os.path.join(save_dir, 'time_complexity_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✅ 保存: time_complexity_table.csv, time_complexity_table.tex")
    print("\n📋 时间复杂度表格:")
    print(df.to_string(index=False))

    return results


# =============================================================================
# 实验5: 混淆矩阵与ROC曲线 (参考 AAFHA Fig.7, Fig.9)
# =============================================================================
def run_confusion_matrix_roc(config, pretrained_path, save_dir):
    """
    混淆矩阵与ROC曲线
    参考AAFHA Fig.7 (ROC) 和 Fig.9 (Confusion Matrix)
    """
    print("\n" + "=" * 60)
    print("实验: Confusion Matrix & ROC Curve (参考AAFHA Fig.7, Fig.9)")
    print("=" * 60)

    set_seed(42)

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    model = MultiModalComplaintModel(
        config=config,
        vocab_size=vocab_size,
        mode='full',
        pretrained_path=pretrained_path
    )

    metrics, all_preds, all_probs, all_targets = quick_train_and_evaluate(
        model, train_loader, val_loader, test_loader, config, num_epochs=10
    )

    print(f"\n性能指标:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")

    # ========== 绘图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    ax = axes[0]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Non-Repeat', 'Repeat'],
        yticklabels=['Non-Repeat', 'Repeat']
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=12)

    # ROC曲线
    ax = axes[1]
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 保存: confusion_matrix_roc.png")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# =============================================================================
# 实验6: 模态语义对齐 (参考假新闻论文 Table 11)
# 维度说明：
#   - text_feat_raw: [batch, 768] (BERT CLS输出)
#   - text_feat: [batch, 256] (通过text_proj投影)
#   - struct_feat: [batch, 256] (通过struct_encoder编码)
#   - 两者维度匹配，可直接计算余弦相似度
# =============================================================================
def run_semantic_alignment(config, pretrained_path, save_dir):
    """
    模态语义对齐分析
    参考假新闻论文Table 11格式：**余弦相似度表格**
    """
    print("\n" + "=" * 60)
    print("实验: Semantic Alignment (参考Table 11)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    model = MultiModalComplaintModel(
        config=config,
        vocab_size=vocab_size,
        mode='full',
        pretrained_path=pretrained_path
    )
    model = model.to(device)
    model.eval()

    # 收集相似度
    similarities_repeat = []
    similarities_non_repeat = []

    def cosine_sim(a, b):
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].numpy()

            # 获取文本特征 (BERT输出)
            # text_output.last_hidden_state: [batch, seq_len, 768]
            # text_feat_raw: [batch, 768] (CLS token)
            text_output = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat_raw = text_output.last_hidden_state[:, 0, :]  # [batch, 768]

            # 投影到256维 (使用模型的text_proj层: 768 -> 256)
            # text_feat: [batch, 256]
            text_feat = model.text_proj(text_feat_raw).cpu().numpy()

            # 获取结构化特征 (通过struct_encoder: 53 -> 256)
            # struct_feat: [batch, 256]
            struct_feat = model.struct_encoder(struct_features).cpu().numpy()

            # 逐样本计算余弦相似度（维度都是256，匹配！）
            for i, target in enumerate(targets):
                sim = cosine_sim(text_feat[i], struct_feat[i])

                if target == 1:
                    similarities_repeat.append(sim)
                else:
                    similarities_non_repeat.append(sim)

    # ========== 生成表格 (Table 11 风格) ==========
    # 使用安全函数处理可能的空列表
    table_data = [
        {
            'Category': 'Repeat Complaint',
            'Mean Similarity': f"{safe_mean(similarities_repeat):.4f}",
            'Std': f"{safe_std(similarities_repeat):.4f}",
            'Min': f"{safe_min(similarities_repeat):.4f}",
            'Max': f"{safe_max(similarities_repeat):.4f}",
            'Count': len(similarities_repeat)
        },
        {
            'Category': 'Non-Repeat Complaint',
            'Mean Similarity': f"{safe_mean(similarities_non_repeat):.4f}",
            'Std': f"{safe_std(similarities_non_repeat):.4f}",
            'Min': f"{safe_min(similarities_non_repeat):.4f}",
            'Max': f"{safe_max(similarities_non_repeat):.4f}",
            'Count': len(similarities_non_repeat)
        }
    ]

    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'semantic_alignment_table.csv'), index=False)

    # LaTeX
    latex_table = df.to_latex(
        index=False,
        caption='Cosine similarity between text and structured features',
        label='tab:semantic_alignment'
    )
    with open(os.path.join(save_dir, 'semantic_alignment_table.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print(f"\n✅ 保存: semantic_alignment_table.csv, semantic_alignment_table.tex")
    print("\n📋 语义对齐表格:")
    print(df.to_string(index=False))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'repeat': {'mean': safe_mean(similarities_repeat), 'std': safe_std(similarities_repeat)},
        'non_repeat': {'mean': safe_mean(similarities_non_repeat), 'std': safe_std(similarities_non_repeat)}
    }


# =============================================================================
# 实验7: LIME可解释性分析 (参考 AAFHA Fig.10-11)
# =============================================================================
# 结构化特征名称（共53个，与config.model.struct_feat_dim=53对应）
STRUCT_FEATURE_NAMES = [
    'Channel', 'Credit', 'Global_Level', 'Upgrade', 'Satisfaction_Time',
    'Urgency_Time', 'Urgency_Accept', 'Transparency', 'Old_User_Online',
    'Policy_Satisfaction', 'New_User_Online', 'New_User_Store', 'Promotion',
    'Network_Satisfaction', 'Performance', 'Service_Usage', 'New_User_Hotline',
    'Expectation', 'Old_User_Hotline', 'Old_User_Store', 'Network_Complaint',
    'NPS_Score', 'Channel_Complaint', 'Other_Complaint', 'No_Complaint',
    'Marketing_Complaint', 'Professionalism', 'Timeliness', 'Result_Satisfaction',
    'Overall_Satisfaction', 'Phone_Status', 'Package_Brand', 'Age', 'Tenure_Months',
    'VIP_Level', 'DND', 'Dual_Card', 'Phone_Brand', 'Campus_User', 'Volte_Potential',
    'Price_Sensitive', 'No_Broadband', 'Competitor_Broadband', 'Card_Apply',
    'Card_Potential', 'Migrant_Worker', 'Other_Return', 'Return_User',
    'Respondent', 'Customer_Segment', 'Gender', 'Feature_52', 'Feature_53'
]  # 共53个

def run_lime_analysis(config, pretrained_path, save_dir):
    """
    LIME可解释性分析
    参考AAFHA Fig.10-11格式：展示top K特征的权重条形图
    """
    print("\n" + "=" * 60)
    print("实验: Integration Analysis / LIME (参考AAFHA Fig.10-11)")
    print("=" * 60)

    set_seed(42)
    device = config.training.device

    train_loader, val_loader, test_loader, vocab_size, processor = prepare_data(config, pretrained_path)

    # 使用单样本batch
    test_loader_single = DataLoader(
        test_loader.dataset,
        batch_size=1,
        collate_fn=custom_collate_fn
    )

    model = MultiModalComplaintModel(
        config=config,
        vocab_size=vocab_size,
        mode='full',
        pretrained_path=pretrained_path
    )
    model = model.to(device)
    model.eval()

    def compute_feature_contributions(batch):
        """计算特征贡献度（扰动法）"""
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        struct_features = batch['struct_features'].to(device)

        # 获取原始预测概率
        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=batch['node_ids'],
                edges_list=batch['edges'],
                node_levels_list=batch['node_levels'],
                struct_features=struct_features
            )
            orig_probs = torch.softmax(logits, dim=1)
            orig_prob = orig_probs[0, 1].item()  # 重复投诉的概率

        contributions = []
        num_features = struct_features.shape[1]  # 应该是53

        # 扰动每个特征，计算贡献度
        for i in range(num_features):
            # 克隆并扰动
            perturbed = struct_features.clone()
            perturbed[0, i] = 0  # 将该特征置零

            with torch.no_grad():
                logits_p, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids_list=batch['node_ids'],
                    edges_list=batch['edges'],
                    node_levels_list=batch['node_levels'],
                    struct_features=perturbed
                )
                new_prob = torch.softmax(logits_p, dim=1)[0, 1].item()

            contribution = orig_prob - new_prob  # 正值表示该特征增加重复投诉概率

            # 获取特征名称
            if i < len(STRUCT_FEATURE_NAMES):
                name = STRUCT_FEATURE_NAMES[i]
            else:
                name = f'Feature_{i}'

            contributions.append((name, contribution))

        # 按贡献度绝对值排序
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        return orig_prob, contributions_sorted

    # 寻找典型案例
    print("\n寻找典型案例...")
    repeat_case = None
    non_repeat_case = None

    for batch in test_loader_single:
        target = batch['target'].item()
        if target == 1 and repeat_case is None:
            orig_prob, contribs = compute_feature_contributions(batch)
            repeat_case = {'prob': orig_prob, 'contributions': contribs}
            print(f"  找到重复投诉案例, prob={orig_prob:.4f}")
        elif target == 0 and non_repeat_case is None:
            orig_prob, contribs = compute_feature_contributions(batch)
            non_repeat_case = {'prob': orig_prob, 'contributions': contribs}
            print(f"  找到非重复投诉案例, prob={orig_prob:.4f}")

        if repeat_case and non_repeat_case:
            break

    # ========== 绘图 (AAFHA Fig.10-11 风格) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cases = [
        (axes[0], repeat_case, 'Repeat Complaint'),
        (axes[1], non_repeat_case, 'Non-Repeat Complaint')
    ]

    for ax, case, title in cases:
        if case is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=12)
            continue

        top_k = 10
        top_features = case['contributions'][:top_k]

        names = [f[0][:15] for f in top_features]  # 截断过长的名称
        values = [f[1] for f in top_features]
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, edgecolor='white')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Contribution Weight', fontsize=11)
        ax.set_title(f'{title}\n(Pred Prob: {case["prob"]:.4f})', fontsize=12)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lime_integration_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 保存: lime_integration_analysis.png")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'repeat': repeat_case, 'non_repeat': non_repeat_case}


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='补充实验脚本（符合参考文献格式）')
    parser.add_argument(
        '--exp',
        type=str,
        default='all',
        choices=['all', 'lr_sensitivity', 'dropout_sensitivity',
                 'fusion', 'time_complexity', 'confusion_matrix',
                 'semantic_alignment', 'lime'],
        help='要运行的实验'
    )
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default='./pretrained_complaint_bert_improved/stage2',
        help='预训练模型路径'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./supplementary_results',
        help='结果保存目录'
    )

    args = parser.parse_args()

    # 确保保存目录存在
    ensure_dir(args.save_dir)

    # 加载配置
    config = Config()

    print("\n" + "=" * 70)
    print("🧪 补充实验脚本（参考文献格式）")
    print("=" * 70)
    print(f"设备: {config.training.device}")
    print(f"保存目录: {args.save_dir}")
    print(f"实验: {args.exp}")
    print("=" * 70)

    # 实验映射
    experiments = {
        'lr_sensitivity': ('学习率敏感性 (AAFHA Fig.3)', run_lr_sensitivity),
        'dropout_sensitivity': ('Dropout敏感性 (AAFHA Fig.4)', run_dropout_sensitivity),
        'fusion': ('融合方式比较 (Table 16/17)', run_fusion_comparison),
        'time_complexity': ('时间复杂度 (AAFHA 4.8节)', run_time_complexity),
        'confusion_matrix': ('混淆矩阵+ROC (AAFHA Fig.7,9)', run_confusion_matrix_roc),
        'semantic_alignment': ('语义对齐 (Table 11)', run_semantic_alignment),
        'lime': ('LIME分析 (AAFHA Fig.10-11)', run_lime_analysis),
    }

    all_results = {}

    # 确定要运行的实验
    if args.exp == 'all':
        exp_list = list(experiments.keys())
    else:
        exp_list = [args.exp]

    # 运行实验
    for exp_name in exp_list:
        if exp_name in experiments:
            desc, func = experiments[exp_name]
            print(f"\n{'=' * 60}")
            print(f"🔬 运行: {desc}")
            print(f"{'=' * 60}")
            try:
                result = func(config, args.pretrained_path, args.save_dir)
                all_results[exp_name] = result
            except Exception as e:
                print(f"❌ 实验失败: {e}")
                import traceback
                traceback.print_exc()

    # 保存所有结果
    def convert_serializable(obj):
        """转换为可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_serializable(i) for i in obj]
        return obj

    results_path = os.path.join(args.save_dir, 'all_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(convert_serializable(all_results), f, ensure_ascii=False, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("✅ 所有实验完成!")
    print("=" * 70)
    print(f"\n📁 输出文件:")
    for filename in sorted(os.listdir(args.save_dir)):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()