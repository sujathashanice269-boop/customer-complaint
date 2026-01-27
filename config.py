"""
配置文件 - 完全改进版
支持六个方向的所有新参数
"""

import json
import os
from typing import Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """模型配置"""
    # BERT相关
    bert_model_name: str = 'bert-base-chinese'
    bert_max_length: int = 256

    # GAT标签编码器
    label_embedding_dim: int = 128
    label_hidden_dim: int = 256
    num_gat_layers: int = 3
    num_gat_heads: int = 4
    max_label_depth: int = 8

    # 跨模态注意力 - 方向四
    use_cross_attention: bool = True
    cross_attn_heads: int = 4

    # 融合层
    fusion_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3

    # 结构化特征 - 方向三
    struct_feat_dim: int = 53
    use_feature_importance: bool = True


@dataclass
class PretrainConfig:
    """预训练配置 - 方向一和方向二"""

    # ===== 方向一: Text预训练 =====
    # 阶段1: 纯MLM领域适应
    stage1_epochs: int = 30
    stage1_lr: float = 5e-5
    stage1_mask_prob: float = 0.15

    # Span Masking策略
    use_span_masking: bool = True
    span_mask_length: int = 3
    span_mask_prob: float = 0.3

    # 阶段2: 对比学习
    stage2_epochs: int = 20
    stage2_lr: float = 3e-5
    use_contrastive: bool = False
    contrastive_temperature: float = 0.5
    contrastive_loss_weight: float = 0.3

    # ===== 方向二: Label全局图预训练 =====
    use_global_graph_pretrain: bool = False
    global_graph_epochs: int = 10
    global_graph_lr: float = 1e-4

    # 全局图预训练任务
    use_node_prediction: bool = True
    use_link_prediction: bool = True
    use_subgraph_classification: bool = True

    # 通用参数
    pretrain_batch_size: int = 32
    save_steps: int = 500
    eval_steps: int = 500


@dataclass
class TrainingConfig:
    """训练配置 - 方向五课程学习"""

    # 数据
    data_file: str = '小案例ai问询.xlsx'
    large_data_file: str = '多模态初始表_数据标签.xlsx'
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    # ===== 方向五: 课程学习策略 =====
    use_curriculum_learning: bool = True

    # 阶段1: 单模态预训练
    stage1_single_modal_epochs: int = 10
    stage1_lr: float = 2e-5

    # 阶段2: 双模态交互
    stage2_dual_modal_epochs: int = 10
    stage2_lr: float = 1e-5

    # 阶段3: 三模态融合
    stage3_full_epochs: int = 20
    stage3_lr: float = 5e-6

    # 如果不使用课程学习
    num_epochs: int = 30
    learning_rate: float = 2e-5

    # 优化器
    batch_size: int = 16
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0

    # 学习率调度
    warmup_steps: int = 500
    scheduler_type: str = 'cosine'

    # 早停
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001

    # ===== 方向六: 模态平衡损失 =====
    use_modal_balance_loss: bool = False
    modal_balance_weight: float = 0.1

    # 损失函数
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # 标签平滑
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1

    # 数据增强
    use_data_augmentation: bool = True
    augmentation_prob: float = 0.5

    # 设备
    device: str = 'cuda'
    num_workers: int = 0

    # 保存和日志
    save_dir: str = './models'
    log_dir: str = './logs'
    save_steps: int = 1000
    log_interval: int = 10

    # 预训练模型路径
    pretrain_save_dir: str = './pretrained_complaint_bert_improved'
    label_pretrain_save_dir: str = './pretrained_label_graph'


@dataclass
class DataConfig:
    """数据配置"""
    max_text_length: int = 256
    max_label_nodes: int = 10

    # 用户词典 - 只用于文本
    user_dict_file: str = 'new_user_dict.txt'

    # 结构化特征列（F列到CR列，共53列）
    struct_start_col: int = 5
    struct_end_col: int = 57

    # 数据增强参数
    text_augment_prob: float = 0.5
    synonym_replace_prob: float = 0.3
    random_delete_prob: float = 0.1
    random_swap_prob: float = 0.1


@dataclass
class Config:
    """完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """初始化后处理"""
        import torch
        if self.training.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA不可用，切换到CPU")
            self.training.device = 'cpu'

        # 创建必要的目录
        os.makedirs(self.training.save_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)
        os.makedirs(self.training.pretrain_save_dir, exist_ok=True)
        os.makedirs(self.training.label_pretrain_save_dir, exist_ok=True)

    def save_config(self, path: str):
        """保存配置"""
        config_dict = {
            'model': asdict(self.model),
            'pretrain': asdict(self.pretrain),
            'training': asdict(self.training),
            'data': asdict(self.data)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已保存到: {path}")

    @classmethod
    def load_config(cls, path: str):
        """加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = cls()
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'pretrain' in config_dict:
            config.pretrain = PretrainConfig(**config_dict['pretrain'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])

        print(f"✅ 配置已加载: {path}")
        return config

    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("配置摘要")
        print("="*60)

        print("\n📊 模型配置:")
        print(f"  - BERT最大长度: {self.model.bert_max_length}")
        print(f"  - GAT层数: {self.model.num_gat_layers}")
        print(f"  - 跨模态注意力: {'✓' if self.model.use_cross_attention else '✗'}")
        print(f"  - 特征重要性加权: {'✓' if self.model.use_feature_importance else '✗'}")
        print(f"  - 结构化特征维度: {self.model.struct_feat_dim}")

        print("\n🎯 预训练配置:")
        print(f"  - Text阶段1 (MLM): {self.pretrain.stage1_epochs}轮")
        print(f"  - Text阶段2 (对比): {self.pretrain.stage2_epochs}轮")
        print(f"  - Span Masking: {'✓' if self.pretrain.use_span_masking else '✗'}")
        print(f"  - 对比学习: {'✓' if self.pretrain.use_contrastive else '✗'}")
        print(f"  - Label全局图预训练: {'✓' if self.pretrain.use_global_graph_pretrain else '✗'}")

        print("\n🚀 训练配置:")
        if self.training.use_curriculum_learning:
            print(f"  - 课程学习: ✓")
            print(f"    • 阶段1 (单模态): {self.training.stage1_single_modal_epochs}轮")
            print(f"    • 阶段2 (双模态): {self.training.stage2_dual_modal_epochs}轮")
            print(f"    • 阶段3 (三模态): {self.training.stage3_full_epochs}轮")
        else:
            print(f"  - 标准训练: {self.training.num_epochs}轮")

        print(f"  - 模态平衡损失: {'✓' if self.training.use_modal_balance_loss else '✗'}")
        print(f"  - Focal Loss: {'✓' if self.training.use_focal_loss else '✗'}")
        print(f"  - 批次大小: {self.training.batch_size}")
        print(f"  - 学习率: {self.training.learning_rate}")
        print(f"  - 设备: {self.training.device}")

        print("\n📁 数据配置:")
        print(f"  - 训练数据: {self.training.data_file}")
        print(f"  - 大规模数据: {self.training.large_data_file}")
        print(f"  - 用户词典: {self.data.user_dict_file}")
        print(f"  - 结构化特征: {self.model.struct_feat_dim}维")

        print("\n💾 保存路径:")
        print(f"  - 模型保存: {self.training.save_dir}")
        print(f"  - Text预训练: {self.training.pretrain_save_dir}")
        print(f"  - Label预训练: {self.training.label_pretrain_save_dir}")
        print("="*60 + "\n")


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def get_quick_test_config() -> Config:
    """获取快速测试配置"""
    config = Config()

    # 减少训练轮数
    config.pretrain.stage1_epochs = 1
    config.pretrain.stage2_epochs = 1
    config.pretrain.global_graph_epochs = 1

    config.training.stage1_single_modal_epochs = 1
    config.training.stage2_dual_modal_epochs = 1
    config.training.stage3_full_epochs = 1
    config.training.num_epochs = 5

    # ⭐ 关键修改5: 调整batch size和学习率
    config.training.batch_size = 8  # 从4改成8,更稳定
    config.pretrain.pretrain_batch_size = 8

    # 降低学习率,特别是对新词
    config.pretrain.stage1_lr = 2e-5  # 从5e-5降低到2e-5
    config.pretrain.stage2_lr = 1e-5  # 从3e-5降低到1e-5

    print("⚡ 使用快速测试配置 (已优化稳定性)")
    return config


def get_production_config() -> Config:
    """获取生产环境配置"""
    config = Config()

    # 完整的预训练轮数
    config.pretrain.stage1_epochs = 30
    config.pretrain.stage2_epochs = 20
    config.pretrain.global_graph_epochs = 20

    # 完整的课程学习
    config.training.use_curriculum_learning = True
    config.training.stage1_single_modal_epochs = 10
    config.training.stage2_dual_modal_epochs = 10
    config.training.stage3_full_epochs = 20

    # 大批量
    config.training.batch_size = 32
    config.pretrain.pretrain_batch_size = 32

    print("🏭 使用生产环境配置")
    return config


if __name__ == "__main__":
    # 测试配置
    print("测试默认配置:")
    config = get_default_config()
    config.print_summary()