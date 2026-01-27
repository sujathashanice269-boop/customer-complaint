"""
消融实验 - 最终修复版本
✅ 修复: 正确加载预训练的全局标签词汇表
✅ 最小改动,保留所有功能
"""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import gc
import json
import os
from tqdm import tqdm

from config import Config
from data_processor import ComplaintDataProcessor, ComplaintDataset, custom_collate_fn
from model import MultiModalComplaintModel


def train_and_evaluate(model, train_loader, val_loader, test_loader, config, device, exp_name):
    """训练和评估模型"""

    # ✅ 修复：针对性训练策略
    # 所有包含TEXT模态的实验都使用分层学习率，保护预训练的BERT参数

    if exp_name in ['text_only', 'text_label', 'text_struct', 'full_model']:
        # 分离 BERT 和 其他参数
        bert_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'text_encoder' in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': 1e-5, 'weight_decay': 0.01},
            {'params': other_params, 'lr': 5e-5}
        ])
        num_epochs = 20
        print(f"✅ {exp_name}: 分层学习率(BERT=1e-5, 其他=5e-5), {num_epochs}轮")

    elif exp_name == 'label_only':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        num_epochs = 20
        print(f"✅ {exp_name}: 学习率1e-4, {num_epochs}轮")

    elif exp_name in ['struct_only', 'label_struct']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        num_epochs = 15
        print(f"✅ {exp_name}: 学习率5e-5, {num_epochs}轮")

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        num_epochs = config.training.num_epochs

    criterion = nn.CrossEntropyLoss()
    best_val_auc = 0
    patience = 0
    max_patience = 3

    for epoch in range(num_epochs):  # ← 改这里
        # 训练
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{config.training.num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_ids_list = batch['node_ids']
            edges_list = batch['edges']
            node_levels_list = batch['node_levels']
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()

            # 根据实验类型选择输入
            if exp_name == 'text_only':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            elif exp_name == 'label_only':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'struct_only':
                logits, _ = model(struct_features=struct_features)
            elif exp_name == 'text_label':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'text_struct':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             struct_features=struct_features)
            elif exp_name == 'label_struct':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)
            else:  # full_model, No_pretrain
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                node_ids_list = batch['node_ids']
                edges_list = batch['edges']
                node_levels_list = batch['node_levels']
                struct_features = batch['struct_features'].to(device)
                targets = batch['target'].to(device)

                # 根据实验类型选择输入
                if exp_name == 'text_only':
                    # ✅ 正确代码
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                elif exp_name == 'label_only':
                    logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list)
                elif exp_name == 'struct_only':
                    logits, _ = model(struct_features=struct_features)
                elif exp_name == 'text_label':
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                 node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list)
                elif exp_name == 'text_struct':
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                 struct_features=struct_features)
                elif exp_name == 'label_struct':
                    logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list, struct_features=struct_features)
                else:
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                 node_ids_list=node_ids_list, edges_list=edges_list,
                                 node_levels_list=node_levels_list, struct_features=struct_features)

                probs = torch.softmax(logits, dim=1)
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_probs)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}")

        # 早停
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"早停触发，最佳验证AUC: {best_val_auc:.4f}")
                break

    # 测试
    model.eval()
    test_preds = []
    test_targets = []
    test_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_ids_list = batch['node_ids']
            edges_list = batch['edges']
            node_levels_list = batch['node_levels']
            struct_features = batch['struct_features'].to(device)
            targets = batch['target'].to(device)

            # 根据实验类型选择输入
            if exp_name == 'text_only':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            elif exp_name == 'label_only':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'struct_only':
                logits, _ = model(struct_features=struct_features)
            elif exp_name == 'text_label':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list)
            elif exp_name == 'text_struct':
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             struct_features=struct_features)
            elif exp_name == 'label_struct':
                logits, _ = model(node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)
            else:
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                             node_ids_list=node_ids_list, edges_list=edges_list,
                             node_levels_list=node_levels_list, struct_features=struct_features)

            probs = torch.softmax(logits, dim=1)
            test_probs.extend(probs[:, 1].cpu().numpy())
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, average='macro')
    auc = roc_auc_score(test_targets, test_probs)

    return accuracy, precision, recall, f1, auc


def run_ablation_study(pretrained_path=None):
    """运行消融实验"""

    print("\n" + "="*60)
    print("消融实验开始")
    print("="*60)

    config = Config()

    # 消融实验用较少轮数
    config.training.num_epochs = 10
    config.training.batch_size = 16

    # 定义实验
    experiments = [
        ('full_model', 'full'),
        ('text_only', 'text_only'),
        ('label_only', 'label_only'),
        ('struct_only', 'struct_only'),
        ('text_label', 'text_label'),
        ('text_struct', 'text_struct'),
        ('label_struct', 'label_struct'),
        ('No_pretrain', 'full'),  # 【新增】真正从零训练
    ]

    results = {}
    # 为每个实验定义独立种子
    experiment_seeds = {
        'full_model': 42,
        'text_only': 43,
        'label_only': 44,
        'struct_only': 45,
        'text_label': 46,
        'text_struct': 47,
        'label_struct': 48,
        'No_pretrain': 50,  # 【新增】
    }
    for exp_name, mode in experiments:
        print(f"\n运行实验: {exp_name}")
        print("-" * 40)
        # 设置该实验的独立随机种子
        seed = experiment_seeds.get(exp_name, 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"✓ 随机种子设置为: {seed}")

        # 决定是否使用预训练
        use_No_pretrain = False  # 【新增】标记是否完全随机初始化BERT


        if exp_name == 'No_pretrain':
            current_pretrained_path = None
            use_No_pretrain = True
            print("[对照组] 完全随机初始化BERT（真正从零训练）")
        else:
            current_pretrained_path = pretrained_path
            if current_pretrained_path and os.path.exists(current_pretrained_path):
                print(f"[实验组] 使用预训练模型: {current_pretrained_path}")
            else:
                print("[警告] 预训练模型路径不存在,将使用原始BERT")

        # ============================================
        # ✅ 关键修改: 创建processor并加载预训练的词汇表
        # ============================================
        processor = ComplaintDataProcessor(
            config=config,
            user_dict_file=config.data.user_dict_file
        )

        # ablation_study.py 第264行附近

        # ✅ 方案1: 优先从父目录加载processor.pkl
        processor_loaded = False
        if current_pretrained_path:
            # pretrained_path = './pretrained_complaint_bert_improved/stage2'
            # parent_dir = './pretrained_complaint_bert_improved'
            parent_dir = os.path.dirname(current_pretrained_path)
            processor_path = os.path.join(parent_dir, 'processor.pkl')

            if os.path.exists(processor_path):
                print(f"✓ 从processor.pkl加载词汇表: {processor_path}")
                processor.load(processor_path)
                processor_loaded = True

        # ✅ 方案2: 如果processor.pkl不存在,尝试加载global_vocab.pkl(新版本)
        if not processor_loaded and current_pretrained_path:
            vocab_path = os.path.join(current_pretrained_path, 'global_vocab.pkl')
            if os.path.exists(vocab_path):
                print(f"✓ 从global_vocab.pkl加载词汇表: {vocab_path}")
                processor.load_global_vocab(vocab_path)
                processor_loaded = True

        # ✅ 如果都没加载成功,prepare_datasets会重新构建(可能导致维度不匹配)
        if not processor_loaded:
            print("⚠️  未找到预训练词汇表,将使用当前数据构建(可能导致维度不匹配)")

        # 加载数据
        print(f"\n加载数据: {config.training.data_file}")

        # ✅ 准备数据集 - 不再传vocab_path,因为已经通过load()加载了
        data = processor.prepare_datasets(
            train_file=config.training.data_file,
            for_pretrain=False
        )

        print(f"数据集大小: {len(data['targets'])}")
        print(f"标签词汇表大小: {data['vocab_size']}")

        # 划分数据集
        total_size = len(data['targets'])
        indices = torch.randperm(total_size).tolist()

        train_size = int(total_size * 0.6)
        val_size = int(total_size * 0.2)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        print(f"训练集: {len(train_indices)}, 验证集: {len(val_indices)}, 测试集: {len(test_indices)}")

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
                'targets': data['targets'][indices]
            }

        train_data = split_data(data, train_indices)
        val_data = split_data(data, val_indices)
        test_data = split_data(data, test_indices)

        # 创建数据集
        train_dataset = ComplaintDataset(
            train_data['text_data'],
            train_data['node_ids_list'],
            train_data['edges_list'],
            train_data['node_levels_list'],
            train_data['struct_features'],
            train_data['targets']
        )

        val_dataset = ComplaintDataset(
            val_data['text_data'],
            val_data['node_ids_list'],
            val_data['edges_list'],
            val_data['node_levels_list'],
            val_data['struct_features'],
            val_data['targets']
        )

        test_dataset = ComplaintDataset(
            test_data['text_data'],
            test_data['node_ids_list'],
            test_data['edges_list'],
            test_data['node_levels_list'],
            test_data['struct_features'],
            test_data['targets']
        )

        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                                  shuffle=True, collate_fn=custom_collate_fn,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size,
                                collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                                 collate_fn=custom_collate_fn)

        # 创建模型
        model = MultiModalComplaintModel(
            config=config,
            vocab_size=data['vocab_size'],
            mode=mode,
            pretrained_path=current_pretrained_path,
            No_pretrain_bert=use_No_pretrain  # 【新增】传递随机初始化标记
        )
        # ✅ 添加这行!
        model = model.to(config.training.device)
        print(f"✓ 模型已移至设备: {config.training.device}")
        # 训练和评估
        accuracy, precision, recall, f1, auc = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            config, config.training.device, exp_name
        )

        results[exp_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        print(f"\n{exp_name} 结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        # 清理内存
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 打印汇总结果
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"\n{'实验名称':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 70)

    for exp_name in ['full_model', 'text_only', 'label_only', 'struct_only',
                     'text_label', 'text_struct', 'label_struct', 'No_pretrain']:
        if exp_name in results:
            r = results[exp_name]
            print(
                f"{exp_name:<15} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['auc']:<10.4f}")

    # 保存结果
    with open('ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("\n结果已保存到: ablation_results.json")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str,
                       default='./pretrained_complaint_bert_improved/stage2',
                       help='预训练模型路径')
    args = parser.parse_args()

    run_ablation_study(args.pretrained_path)