"""
多模态投诉预测模型 - 完全改进版
实现六个方向的所有改进
✅ 修复：text_only模式的维度不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import numpy as np
import os


# ============================================================
# 新增: BERT对比学习模型
# ============================================================

class BERTForContrastiveLearning(nn.Module):
    """BERT + Projection Head for Supervised Contrastive Learning"""

    def __init__(self, bert_model_name='bert-base-chinese', projection_dim=128):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # Projection Head: 768 → 256 → 128
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, input_ids, attention_mask, return_projection=True):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch, 768]

        if return_projection:
            projection = self.projection(pooled_output)
            features = F.normalize(projection, dim=1)  # L2归一化
            return features
        else:
            return pooled_output

    def get_bert_only(self):
        """提取BERT (丢弃projection)"""
        return self.bert


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch, dim] 归一化特征
            labels: [batch] 类别标签
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. 构造标签mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 2. 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)

        # 3. 去掉对角线
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # 4. 温度缩放
        logits = similarity_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 5. 计算log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 6. 计算loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss


class CrossModalAttention(nn.Module):
    """跨模态注意力模块 - 方向四核心创新"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, query, key_value):
        """
        Args:
            query: [batch, 1, dim]
            key_value: [batch, 1, dim]
        Returns:
            enhanced: [batch, 1, dim]
            attn_weights: 注意力权重
        """
        # Cross-Attention
        attn_output, attn_weights = self.attention(query, key_value, key_value)

        # 残差连接
        query = self.layer_norm1(query + attn_output)

        # FFN
        ffn_output = self.ffn(query)
        output = self.layer_norm2(query + ffn_output)

        return output, attn_weights


class TextMultiTokenGenerator(nn.Module):
    """从BERT多层输出生成多个语义Token"""

    def __init__(self, bert_hidden_size=768, output_dim=256, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bert_hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for _ in range(num_tokens)
        ])

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

    def forward(self, bert_hidden_states):
        """
        Args:
            bert_hidden_states: tuple，BERT所有层输出
        Returns:
            text_tokens: [batch, num_tokens, output_dim]
        """
        batch_size = bert_hidden_states[-1].size(0)

        tokens = []
        for i, proj in enumerate(self.layer_projections):
            layer_idx = -(self.num_tokens - i)  # -4, -3, -2, -1
            cls_token = bert_hidden_states[layer_idx][:, 0, :]  # [batch, 768]
            projected = proj(cls_token)  # [batch, 256]
            tokens.append(projected.unsqueeze(1))

        text_tokens = torch.cat(tokens, dim=1)  # [batch, 4, 256]
        text_tokens = text_tokens + self.position_embeddings.expand(batch_size, -1, -1)

        return text_tokens


class StructMultiTokenGenerator(nn.Module):
    """将结构化特征生成多个Token"""

    def __init__(self, input_dim=53, output_dim=256, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

        self.token_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_tokens)
        ])

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

    def forward(self, struct_features):
        """
        Args:
            struct_features: [batch, input_dim]
        Returns:
            struct_tokens: [batch, num_tokens, output_dim]
        """
        batch_size = struct_features.size(0)

        tokens = []
        for generator in self.token_generators:
            token = generator(struct_features)  # [batch, 256]
            tokens.append(token.unsqueeze(1))

        struct_tokens = torch.cat(tokens, dim=1)  # [batch, 4, 256]
        struct_tokens = struct_tokens + self.position_embeddings.expand(batch_size, -1, -1)

        return struct_tokens


class TextLedCrossModalAttention(nn.Module):
    """
    文本主导的跨模态注意力

    设计：
    - 文本对标签和结构做跨模态注意力
    - 各模态自注意力增强表示
    - 门控融合，文本权重更高
    """

    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 文本对其他模态的注意力
        self.text_to_label_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_struct_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # 各模态自注意力
        self.text_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.label_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.struct_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # 层归一化
        self.text_norm = nn.LayerNorm(dim)
        self.label_norm = nn.LayerNorm(dim)
        self.struct_norm = nn.LayerNorm(dim)

        # 门控融合
        self.modal_gate = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

        # 文本偏置（确保文本权重更高）
        self.text_bias = nn.Parameter(torch.tensor(0.5))
        # 可学习的跨模态融合权重
        self.cross_modal_weight_label = nn.Parameter(torch.tensor(0.0))
        self.cross_modal_weight_struct = nn.Parameter(torch.tensor(0.0))

    def forward(self, text_tokens, label_tokens, struct_tokens,
                label_mask=None, return_attention=True):
        """
        Args:
            text_tokens: [batch, 4, 256]
            label_tokens: [batch, N, 256]
            struct_tokens: [batch, 4, 256]
            label_mask: [batch, N] - True表示padding位置
        """
        attention_weights = {}

        # 1. 自注意力
        text_self, _ = self.text_self_attn(text_tokens, text_tokens, text_tokens)
        text_tokens = self.text_norm(text_tokens + text_self)

        label_self, attn_l = self.label_self_attn(
            label_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )
        label_tokens = self.label_norm(label_tokens + label_self)

        struct_self, _ = self.struct_self_attn(struct_tokens, struct_tokens, struct_tokens)
        struct_tokens = self.struct_norm(struct_tokens + struct_self)

        # 2. 文本主导的跨模态注意力
        text_to_label, attn_t2l = self.text_to_label_attn(
            text_tokens, label_tokens, label_tokens,
            key_padding_mask=label_mask,
            need_weights=return_attention,
            average_attn_weights=False
        )

        text_to_struct, attn_t2s = self.text_to_struct_attn(
            text_tokens, struct_tokens, struct_tokens,
            need_weights=return_attention,
            average_attn_weights=False
        )

        # 3. 特征增强
        weight_label = torch.sigmoid(self.cross_modal_weight_label)
        weight_struct = torch.sigmoid(self.cross_modal_weight_struct)
        text_enhanced = text_tokens + weight_label * text_to_label + weight_struct * text_to_struct
        label_enhanced = label_tokens
        struct_enhanced = struct_tokens

        # 4. 池化
        text_pooled = text_enhanced.mean(dim=1)  # [batch, 256]
        label_pooled = label_enhanced.mean(dim=1)  # [batch, 256]
        struct_pooled = struct_enhanced.mean(dim=1)  # [batch, 256]

        # 5. 门控融合（文本主导）
        gate_input = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
        gate_logits = self.modal_gate(gate_input)  # [batch, 3]
        gate_logits[:, 0] = gate_logits[:, 0] + self.text_bias  # 文本偏置
        gate_weights = F.softmax(gate_logits, dim=-1)  # [batch, 3]

        if return_attention:
            attention_weights = {
                'text_to_label': attn_t2l,  # [batch, heads, 4, N]
                'text_to_struct': attn_t2s,  # [batch, heads, 4, 4]
                'label_self': attn_l,  # [batch, heads, N, N]
                'modal_weights': gate_weights,  # [batch, 3]
            }

        return text_pooled, label_pooled, struct_pooled, attention_weights


class GATLabelEncoder(nn.Module):
    """GAT标签编码器 - 支持全局图预训练"""

    def __init__(self, vocab_size, hidden_dim=256, num_layers=3, num_heads=4, max_level=8):
        super().__init__()

        # 节点嵌入
        self.node_embedding = nn.Embedding(vocab_size, 128)

        # 层级嵌入 - 方向二改进
        self.level_embedding = nn.Embedding(max_level + 1, 32)

        # 层级权重 - 使用递增初始化，让深层节点有更大初始权重
        level_init = torch.zeros(max_level + 1)
        for i in range(max_level + 1):
            level_init[i] = i * 0.1  # 深层权重更大
        self.level_weights = nn.Parameter(level_init)

        # GAT层
        self.gat_layers = nn.ModuleList()
        input_dim = 128 + 32

        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.3)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.3)
                )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_ids, edge_index, node_levels, batch=None):
        """前向传播"""
        # 节点嵌入
        x = self.node_embedding(node_ids)  # [num_nodes, 128]

        # 层级嵌入
        level_emb = self.level_embedding(node_levels)  # [num_nodes, 32]

        # 拼接
        x = torch.cat([x, level_emb], dim=-1)  # [num_nodes, 160]

        # GAT层
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)

        # 层级加权 - 方向二改进
        level_weights = torch.softmax(self.level_weights, dim=0)
        weighted_x = x * level_weights[node_levels].unsqueeze(-1)

        # 输出投影和归一化（对所有节点）
        weighted_x = self.output_proj(weighted_x)
        weighted_x = self.layer_norm(weighted_x)

        # ========== 返回所有节点特征（修复P01）==========
        max_nodes = 8  # 标签路径最大节点数

        if batch is not None:
            batch_features = []
            batch_masks = []
            unique_batches = torch.unique(batch)

            for b in unique_batches:
                node_mask = (batch == b)
                nodes = weighted_x[node_mask]  # [num_nodes, hidden_dim]
                num_nodes = nodes.size(0)

                # Padding到max_nodes
                if num_nodes < max_nodes:
                    pad_size = max_nodes - num_nodes
                    padding = torch.zeros(pad_size, nodes.size(-1), device=nodes.device)
                    nodes = torch.cat([nodes, padding], dim=0)
                    attn_mask = torch.cat([
                        torch.zeros(num_nodes, dtype=torch.bool, device=nodes.device),
                        torch.ones(pad_size, dtype=torch.bool, device=nodes.device)
                    ])
                else:
                    nodes = nodes[:max_nodes]
                    attn_mask = torch.zeros(max_nodes, dtype=torch.bool, device=nodes.device)

                batch_features.append(nodes.unsqueeze(0))
                batch_masks.append(attn_mask.unsqueeze(0))

            node_features = torch.cat(batch_features, dim=0)  # [batch, max_nodes, dim]
            node_masks = torch.cat(batch_masks, dim=0)  # [batch, max_nodes]
            return node_features, node_masks

        # 单样本情况
        num_nodes = weighted_x.size(0)
        if num_nodes < max_nodes:
            pad_size = max_nodes - num_nodes
            padding = torch.zeros(pad_size, weighted_x.size(-1), device=weighted_x.device)
            weighted_x = torch.cat([weighted_x, padding], dim=0)
            attn_mask = torch.cat([
                torch.zeros(num_nodes, dtype=torch.bool, device=weighted_x.device),
                torch.ones(pad_size, dtype=torch.bool, device=weighted_x.device)
            ])
        else:
            weighted_x = weighted_x[:max_nodes]
            attn_mask = torch.zeros(max_nodes, dtype=torch.bool, device=weighted_x.device)

        return weighted_x.unsqueeze(0), attn_mask.unsqueeze(0)


# ============================================================
# 新增: Label回归预训练模块
# ============================================================

class LabelRiskRegressor(nn.Module):
    """
    标签路径风险回归器

    目标: 预测给定标签路径的重复投诉率 (0-1之间的连续值)

    架构:
        GATLabelEncoder → PathEmbedding [256维] → Regressor → Risk Score [1维]

    训练目标:
        给定标签路径 L1→L2→L3，预测该路径的历史重复投诉率
        例如: "服务质量→响应速度→超时" → 0.23 (23%重复率)
    """

    def __init__(self, label_encoder, hidden_dim=256):
        """
        Args:
            label_encoder: 已有的GATLabelEncoder
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.label_encoder = label_encoder

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1之间
        )

    def forward(self, node_ids, edge_index, node_levels, batch=None):
        """
        前向传播

        Args:
            node_ids: 节点ID
            edge_index: 边索引
            node_levels: 节点层级
            batch: batch索引

        Returns:
            risk_scores: [batch_size, 1] 预测的风险分数
        """
        # 获取标签路径嵌入
        label_output = self.label_encoder(node_ids, edge_index, node_levels, batch)

        # 处理新的返回格式（tuple）
        if isinstance(label_output, tuple):
            label_embedding = label_output[0].mean(dim=1)  # [batch, dim]
        else:
            label_embedding = label_output  # [batch, dim]

        # 回归预测
        risk_scores = self.regressor(label_embedding)
        return risk_scores

    def compute_loss(self, predicted, target):
        """计算MSE损失"""
        return F.mse_loss(predicted.squeeze(), target.float())


# ============================================================
# 新增: 多任务预训练模块
# ============================================================

class MultiTaskPretrainer(nn.Module):
    """
    多任务预训练: 同时进行文本对比学习 + 标签回归

    设计理念:
        - 文本编码器通过对比学习获得更好的语义表示
        - 标签编码器通过回归任务学习标签路径的风险模式
        - 两者独立预训练，然后在主模型中融合
    """

    def __init__(self, text_model, label_regressor, text_weight=0.5, label_weight=0.5, temperature=0.5):
        super().__init__()
        self.text_model = text_model
        self.label_regressor = label_regressor
        self.text_weight = text_weight
        self.label_weight = label_weight
        self.temperature = temperature
        self.contrastive_loss = SupConLoss(temperature=self.temperature)

    def forward(self, text_inputs, label_inputs, text_labels, label_targets):
        """
        前向传播

        Args:
            text_inputs: (input_ids, attention_mask)
            label_inputs: (node_ids, edge_index, node_levels, batch)
            text_labels: 文本类别标签
            label_targets: 标签路径风险分数

        Returns:
            total_loss, text_loss, label_loss
        """
        # 文本对比学习
        input_ids, attention_mask = text_inputs
        text_features = self.text_model(input_ids, attention_mask)
        text_loss = self.contrastive_loss(text_features, text_labels)

        # 标签回归
        node_ids, edge_index, node_levels, batch = label_inputs
        risk_pred = self.label_regressor(node_ids, edge_index, node_levels, batch)
        label_loss = self.label_regressor.compute_loss(risk_pred, label_targets)

        # 加权总损失
        total_loss = self.text_weight * text_loss + self.label_weight * label_loss

        return total_loss, text_loss, label_loss


# ============================================================
# 新增: 全局图预训练损失
# ============================================================

class GlobalGraphPretrainLoss(nn.Module):
    """
    全局图预训练损失

    包含三个子任务:
    1. 节点分类: 预测节点的层级
    2. 边预测: 预测两个节点之间是否存在边
    3. 图对比学习: 增强图的表示能力
    """

    def __init__(self, hidden_dim=256, num_levels=8):
        super().__init__()

        # 节点分类头
        self.level_classifier = nn.Linear(hidden_dim, num_levels + 1)

        # 边预测头
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, node_embeddings, edge_index, node_levels, batch=None):
        """
        计算预训练损失

        Args:
            node_embeddings: [num_nodes, hidden_dim] 节点嵌入
            edge_index: [2, num_edges] 边索引
            node_levels: [num_nodes] 节点层级
            batch: [num_nodes] batch索引

        Returns:
            total_loss, level_loss, edge_loss
        """
        # 1. 节点层级分类损失
        level_logits = self.level_classifier(node_embeddings)
        level_loss = F.cross_entropy(level_logits, node_levels)

        # 2. 边预测损失
        # 正样本: 实际存在的边
        src, dst = edge_index
        pos_pairs = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
        pos_scores = self.edge_predictor(pos_pairs).squeeze()
        pos_labels = torch.ones_like(pos_scores)

        # 负样本: 随机采样不存在的边
        num_neg = edge_index.size(1)
        num_nodes = node_embeddings.size(0)
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=edge_index.device)
        neg_pairs = torch.cat([node_embeddings[neg_src], node_embeddings[neg_dst]], dim=-1)
        neg_scores = self.edge_predictor(neg_pairs).squeeze()
        neg_labels = torch.zeros_like(neg_scores)

        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        edge_loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)

        # 总损失
        total_loss = level_loss + edge_loss

        return total_loss, level_loss, edge_loss


# ============================================================
# 新增: 标签预训练 - 共现预测
# ============================================================

class LabelCooccurrenceLoss(nn.Module):
    """
    标签共现预测损失

    任务: 给定一个标签路径，预测哪些其他标签经常一起出现
    这有助于学习标签之间的语义关系

    训练数据格式:
        输入: 标签路径 L1→L2→L3 的嵌入
        目标: 多标签分类，预测相关标签
    """

    def __init__(self, hidden_dim=256, num_labels=1000):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, label_embedding, cooccur_labels):
        """
        Args:
            label_embedding: [batch, hidden_dim] 标签路径嵌入
            cooccur_labels: [batch, num_labels] 多热编码的共现标签

        Returns:
            loss: BCE损失
        """
        logits = self.predictor(label_embedding)
        loss = F.binary_cross_entropy_with_logits(logits, cooccur_labels.float())
        return loss


# ============================================================
# 另一种对比学习实现 - 更灵活的版本
# ============================================================

class FlexibleSupConLoss(nn.Module):
    """
    灵活的监督对比学习损失

    支持:
    - 批内对比
    - 多正样本对比
    - 温度可调
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        计算监督对比损失

        Args:
            features: [batch, n_views, dim] 或 [batch, dim]
            labels: [batch]

        Returns:
            对比损失
        """
        device = features.device

        # 如果是2D，添加view维度
        if features.dim() == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # 4.1 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )

        # 4.2 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 4.3 构建mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(contrast_count, contrast_count)

        # 4.4 排除自身
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # 4.5 计算log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 4.6 最终loss: 负的平均log-prob
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss


class MultiModalComplaintModel(nn.Module):
    """多模态投诉预测模型 - 完整版"""

    def __init__(self, config, vocab_size, mode='full', pretrained_path=None, No_pretrain_bert=False):
        """
        Args:
            config: 配置对象
            vocab_size: 标签词汇表大小
            mode: 模型模式
                - 'full': 完整三模态
                - 'text_only': 仅文本
                - 'label_only': 仅标签
                - 'struct_only': 仅结构化
                - 'text_label': 文本+标签
                - 'text_struct': 文本+结构化
                - 'label_struct': 标签+结构化
            pretrained_path: 预训练模型路径
        """
        super().__init__()

        self.config = config
        self.mode = mode
        self.device = config.training.device

        # ========== 文本编码器 (BERT) - 方向一改进 ==========
        if mode in ['full', 'text_only', 'text_label', 'text_struct']:
            if No_pretrain_bert:
                # 【新增】完全随机初始化BERT（真正从零训练）
                print("🔄 使用随机初始化的BERT（从零训练）")
                from transformers import BertConfig
                bert_config = BertConfig.from_pretrained(config.model.bert_model_name)
                self.text_encoder = BertModel(bert_config)  # 随机初始化权重
            elif pretrained_path and os.path.exists(pretrained_path):
                print(f"✅ 加载领域预训练BERT: {pretrained_path}")
                self.text_encoder = BertModel.from_pretrained(pretrained_path)
            else:
                print("📦 使用原始BERT预训练权重（无领域预训练）")
                self.text_encoder = BertModel.from_pretrained(config.model.bert_model_name)

            # ✅ 核心修复：添加投影层（统一维度到256）
            self.text_proj = nn.Linear(768, 256)

            # 对比学习投影头（如果需要）
            self.text_contrast_proj = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        else:
            self.text_encoder = None
            self.text_proj = None  # ✅ 明确设置为None
            self.text_contrast_proj = None

        # ========== 标签编码器 (GAT) - 方向二改进 ==========
        if mode in ['full', 'label_only', 'text_label', 'label_struct']:
            self.label_encoder = GATLabelEncoder(
                vocab_size=vocab_size,
                hidden_dim=256,
                num_layers=3,
                num_heads=4
            )

            # 加载全局图预训练权重
            if pretrained_path:
                label_pretrain_path = os.path.join(
                    config.training.label_pretrain_save_dir,
                    'label_global_pretrain.pth'
                )
                if os.path.exists(label_pretrain_path):
                    try:
                        state_dict = torch.load(label_pretrain_path, map_location=self.device)
                        self.label_encoder.load_state_dict(state_dict, strict=False)
                        print("✅ 加载全局图预训练权重")
                    except Exception as e:
                        print(f"⚠️ 全局图预训练权重加载失败: {e}")
        else:
            self.label_encoder = None

        # ========== 结构化特征编码器 - 方向三改进 ==========
        if mode in ['full', 'struct_only', 'text_struct', 'label_struct']:
            assert config.model.struct_feat_dim == 53, \
                f"结构化特征必须是53维，当前{config.model.struct_feat_dim}维"

            self.struct_encoder = nn.Sequential(
                nn.Linear(53, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            # 特征重要性权重 - 使用正态分布初始化以打破对称性
            self.feature_importance = nn.Parameter(torch.randn(53) * 0.1)
        else:
            self.struct_encoder = None
            self.feature_importance = None

        # ========== 跨模态注意力 - 文本主导方案（修复P01）==========
        if mode == 'full':
            # Token生成器
            self.text_token_gen = TextMultiTokenGenerator(
                bert_hidden_size=768, output_dim=256, num_tokens=4
            )
            self.struct_token_gen = StructMultiTokenGenerator(
                input_dim=53, output_dim=256, num_tokens=4
            )
            # 文本主导的跨模态注意力
            self.text_led_cross_modal = TextLedCrossModalAttention(
                dim=256, num_heads=4, dropout=0.1
            )
        elif mode in ['text_label', 'text_struct', 'label_struct']:
            # 双模态交互
            self.modal_attn_1 = CrossModalAttention(256, num_heads=4)
            self.modal_attn_2 = CrossModalAttention(256, num_heads=4)

        # ========== 融合层 ==========
        # 计算融合层输入维度
        if mode == 'full':
            fusion_input_dim = 256 * 3  # text + label + struct
        elif mode in ['text_label', 'text_struct', 'label_struct']:
            fusion_input_dim = 256 * 2
        else:
            fusion_input_dim = 256  # 单模态统一256维

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.model.fusion_dim),
            nn.LayerNorm(config.model.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.fusion_dim, config.model.hidden_dim),
            nn.LayerNorm(config.model.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        )

        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.model.hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        # ========== 源头预防: 权重初始化 ==========
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化模型权重 - 源头预防数值问题
        使用Xavier初始化确保训练稳定
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier均匀初始化 (适合tanh/sigmoid激活)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.MultiheadAttention):
                # 注意力层的特殊初始化
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)

            elif isinstance(module, nn.Embedding):
                # 嵌入层正态初始化
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.Parameter):
                # 参数初始化 (如feature_importance)
                if module.dim() > 0:
                    nn.init.ones_(module)

        print("✅ 模型权重初始化完成")

    def forward(self, input_ids=None, attention_mask=None,
                node_ids_list=None, edges_list=None, node_levels_list=None,
                struct_features=None, return_attention=False):
        """
        前向传播
        Args:
            input_ids: 文本输入 [batch, seq_len]
            attention_mask: 注意力mask [batch, seq_len]
            node_ids_list: 节点ID列表 (list of lists)
            edges_list: 边列表 (list of lists)
            node_levels_list: 节点层级列表 (list of lists)
            struct_features: 结构化特征 [batch, 53]
            return_attention: 是否返回注意力权重
        Returns:
            logits: 分类logits [batch, 2]
            attention_weights: 注意力权重字典（如果return_attention=True）
        """
        attention_weights = {}

        # ========== 文本特征 ==========
        bert_hidden_states = None  # 新增：保存hidden_states用于多Token生成
        if self.text_encoder is not None and input_ids is not None:
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 保存所有层的hidden_states（用于多Token生成）
            bert_hidden_states = text_output.hidden_states  # tuple of [batch, seq, 768]

            # 原有逻辑保持（用于非full模式的兼容）
            text_feat = text_output.last_hidden_state[:, 0, :]  # [batch, 768]

            if self.text_proj is not None:
                text_feat_proj = self.text_proj(text_feat)  # [batch, 256]
                text_feat_proj = text_feat_proj.unsqueeze(1)  # [batch, 1, 256]
            else:
                raise ValueError("text_proj不应为None！")
        else:
            text_feat_proj = None
            bert_hidden_states = None

        # ========== 标签特征 ==========
        if self.label_encoder is not None and node_ids_list is not None:
            batch_data = []
            for i in range(len(node_ids_list)):
                node_ids = torch.tensor(node_ids_list[i], dtype=torch.long, device=self.device)
                node_levels = torch.tensor(node_levels_list[i], dtype=torch.long, device=self.device)

                # 构建边
                if edges_list[i]:
                    edges = torch.tensor(edges_list[i], dtype=torch.long, device=self.device).t()
                else:
                    # 自环
                    num_nodes = len(node_ids)
                    edges = torch.tensor([[j, j] for j in range(num_nodes)], device=self.device).t()

                data = Data(
                    x=node_ids,
                    edge_index=edges,
                    node_levels=node_levels,
                    batch=torch.full((len(node_ids),), i, dtype=torch.long, device=self.device)
                )
                batch_data.append(data)

            graph_batch = Batch.from_data_list(batch_data).to(self.device)
            # 获取标签多节点特征（修复P01）
            label_feat = self.label_encoder(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.node_levels,
                graph_batch.batch
            )  # 返回 (node_features, node_masks)
            # 注意：label_feat现在是tuple，在跨模态交互中处理
        else:
            label_feat = None

        # ========== 结构化特征 ==========
        if self.struct_encoder is not None and struct_features is not None:
            # 特征重要性加权 - 方向三
            if hasattr(self, 'feature_importance'):
                importance_weights = torch.softmax(self.feature_importance, dim=0)
                struct_features = struct_features * importance_weights

            struct_feat = self.struct_encoder(struct_features)  # [batch, 256]
            struct_feat = struct_feat.unsqueeze(1)  # [batch, 1, 256]
        else:
            struct_feat = None

        # ========== 跨模态交互 - 文本主导方案 ==========
        if self.mode == 'full':
            if text_feat_proj is not None and label_feat is not None and struct_feat is not None:
                # 处理标签返回值（现在是多节点版本）
                if isinstance(label_feat, tuple):
                    label_node_feats, label_mask = label_feat
                else:
                    # 兼容旧版本
                    label_node_feats = label_feat
                    label_mask = None

                # 生成文本多Token
                if bert_hidden_states is not None:
                    text_tokens = self.text_token_gen(bert_hidden_states)  # [batch, 4, 256]
                else:
                    # 回退：复制text_feat_proj
                    text_tokens = text_feat_proj.expand(-1, 4, -1)

                # 生成结构多Token
                struct_tokens = self.struct_token_gen(struct_features)  # [batch, 4, 256]

                # 文本主导的跨模态注意力
                text_pooled, label_pooled, struct_pooled, cross_attn = \
                    self.text_led_cross_modal(
                        text_tokens, label_node_feats, struct_tokens,
                        label_mask=label_mask,
                        return_attention=return_attention
                    )

                if return_attention and cross_attn:
                    attention_weights.update(cross_attn)

                # 拼接三个模态的池化特征
                combined_feat = torch.cat([text_pooled, label_pooled, struct_pooled], dim=-1)
            else:
                # 回退：部分模态缺失
                features = []
                if text_feat_proj is not None:
                    features.append(text_feat_proj.squeeze(1))
                if label_feat is not None:
                    if isinstance(label_feat, tuple):
                        features.append(label_feat[0].mean(dim=1))
                    else:
                        features.append(label_feat.squeeze(1))
                if struct_feat is not None:
                    features.append(struct_feat.squeeze(1))
                combined_feat = torch.cat(features, dim=-1)

        elif self.mode in ['text_label', 'text_struct', 'label_struct']:
            # 双模态交互
            feat1, feat2 = None, None

            if self.mode == 'text_label':
                feat1 = text_feat_proj
                # 修复: label_encoder返回的是元组(node_features, node_masks)
                if isinstance(label_feat, tuple):
                    node_feats, node_mask = label_feat
                    if node_mask is not None:
                        valid_mask = ~node_mask
                        mask_expanded = valid_mask.unsqueeze(-1).float()
                        label_pooled = (node_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
                    else:
                        label_pooled = node_feats.mean(dim=1)
                    feat2 = label_pooled.unsqueeze(1)
                else:
                    feat2 = label_feat
            elif self.mode == 'text_struct':
                feat1, feat2 = text_feat_proj, struct_feat
            elif self.mode == 'label_struct':
                # 修复: label_encoder返回的是元组(node_features, node_masks)
                if isinstance(label_feat, tuple):
                    node_feats, node_mask = label_feat
                    if node_mask is not None:
                        valid_mask = ~node_mask
                        mask_expanded = valid_mask.unsqueeze(-1).float()
                        label_pooled = (node_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
                    else:
                        label_pooled = node_feats.mean(dim=1)
                    feat1 = label_pooled.unsqueeze(1)
                else:
                    feat1 = label_feat
                feat2 = struct_feat

            if feat1 is not None and feat2 is not None:
                feat1_enhanced, attn1 = self.modal_attn_1(feat1, feat2)
                feat2_enhanced, attn2 = self.modal_attn_2(feat2, feat1)

                if return_attention:
                    attention_weights['modal1_to_modal2'] = attn1
                    attention_weights['modal2_to_modal1'] = attn2

                combined_feat = torch.cat([
                    feat1_enhanced.squeeze(1),
                    feat2_enhanced.squeeze(1)
                ], dim=-1)
            else:
                # 如果某个模态缺失，使用可用的
                available = [f for f in [feat1, feat2] if f is not None]
                if available:
                    combined_feat = torch.cat([f.squeeze(1) for f in available], dim=-1)
                else:
                    raise ValueError("至少需要一个模态的输入")

        else:
            # 单模态
            if text_feat_proj is not None:
                combined_feat = text_feat_proj.squeeze(1)
            elif label_feat is not None:
                if isinstance(label_feat, tuple):
                    node_feats, node_mask = label_feat
                    if node_mask is not None:
                        valid_mask = ~node_mask  # True表示padding，取反得到有效位置
                        mask_expanded = valid_mask.unsqueeze(-1).float()
                        combined_feat = (node_feats * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
                    else:
                        combined_feat = node_feats.mean(dim=1)
                else:
                    combined_feat = label_feat.squeeze(1)
            elif struct_feat is not None:
                combined_feat = struct_feat.squeeze(1)
            else:
                raise ValueError("至少需要一个模态的输入")

        # ========== 融合和分类 ==========
        fused_feat = self.fusion(combined_feat)  # [batch, hidden_dim]
        logits = self.classifier(fused_feat)  # [batch, 2]

        # ========== 源头预防: 统一返回格式 ==========
        if return_attention:
            # 确保attention_weights不为空
            if not attention_weights:
                attention_weights = self._get_default_attention_weights()
            return logits, attention_weights
        else:
            # 即使不需要attention，也返回None保持格式统一
            return logits, None

    def _get_default_attention_weights(self):
        """获取默认的注意力权重 - 使用有意义的初始化"""
        max_nodes = 8
        num_heads = 4
        text_seq_len = 4
        struct_seq_len = 4

        # 文本对标签: 递减权重（前面的标签节点更重要）
        t2l_weights = torch.zeros(1, num_heads, text_seq_len, max_nodes, device=self.device)
        for i in range(max_nodes):
            t2l_weights[:, :, :, i] = 1.0 / (i + 1)
        t2l_weights = t2l_weights / t2l_weights.sum(dim=-1, keepdim=True)

        # 文本对结构: 递减权重
        t2s_weights = torch.zeros(1, num_heads, text_seq_len, struct_seq_len, device=self.device)
        for i in range(struct_seq_len):
            t2s_weights[:, :, :, i] = 1.0 / (i + 1)
        t2s_weights = t2s_weights / t2s_weights.sum(dim=-1, keepdim=True)

        # 标签自注意力: 对角线强调（自注意力更强）
        l_self_weights = torch.zeros(1, num_heads, max_nodes, max_nodes, device=self.device)
        for i in range(max_nodes):
            l_self_weights[:, :, i, i] = 0.5  # 对角线权重
            for j in range(max_nodes):
                if i != j:
                    l_self_weights[:, :, i, j] = 0.5 / (max_nodes - 1)

        # 门控权重: 文本主导
        gate_weights = torch.tensor([[0.5, 0.25, 0.25]], device=self.device)

        return {
            'text_to_label': t2l_weights,
            'text_to_struct': t2s_weights,
            'label_self': l_self_weights,
            'modal_weights': gate_weights,
        }


class ThresholdCalibrator:
    """
    阈值校准器

    作用: 在预训练后，根据验证集统计最优分类阈值
    """

    def __init__(self):
        self.threshold = 0.5  # 默认阈值
        self.statistics = {}

    def calibrate(self, predicted_risks, true_labels):
        """
        校准阈值

        Args:
            predicted_risks: 预测的风险分数 [N]
            true_labels: 真实标签 [N], 0或1

        Returns:
            optimal_threshold: 最优阈值
        """
        from sklearn.metrics import roc_curve, auc

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_risks)

        # Youden指数: J = TPR - FPR
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]

        self.threshold = optimal_threshold
        self.statistics = {
            'threshold': optimal_threshold,
            'tpr': tpr[optimal_idx],
            'fpr': fpr[optimal_idx],
            'auc': auc(fpr, tpr)
        }

        print(f"\n📊 阈值校准完成:")
        print(f"   最优阈值: {optimal_threshold:.4f}")
        print(f"   TPR: {tpr[optimal_idx]:.4f}")
        print(f"   FPR: {fpr[optimal_idx]:.4f}")
        print(f"   AUC: {auc(fpr, tpr):.4f}")

        return optimal_threshold

    def predict(self, risk_scores):
        """
        使用校准后的阈值进行预测

        Args:
            risk_scores: 风险分数 [N]

        Returns:
            predictions: 预测标签 [N], 0或1
        """
        return (risk_scores >= self.threshold).astype(int)


class FocalLoss(nn.Module):
    """Focal Loss - 方向六"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch, num_classes]
            targets: [batch]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss - 处理类别不平衡"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """组合损失函数"""

    def __init__(self, focal_weight=0.5, dice_weight=0.3, ce_weight=0.2):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        ce_loss = F.cross_entropy(inputs, targets)

        return (self.focal_weight * focal_loss +
                self.dice_weight * dice_loss +
                self.ce_weight * ce_loss)


class ModalBalanceLoss(nn.Module):
    """
    模态平衡损失 - 方向六（改进版）
    基于注意力权重的熵最大化，强制三个模态贡献均衡

    原理：
    1. 从跨模态注意力中提取每个模态的贡献度权重
    2. 计算权重分布的熵
    3. 熵越大，分布越均匀，模态越平衡
    4. 通过最大化熵，强制各模态平衡贡献

    效果：
    - 防止Struct主导
    - 让Text关注文本细节
    - 让Label关注类别信息
    - 让Struct关注客户画像
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        self.epsilon = 1e-8  # 防止log(0)

    def forward(self, attention_weights_dict):
        """
        计算基于注意力权重的平衡损失

        Args:
            attention_weights_dict: 注意力权重字典，格式：
                {
                    'text_to_label': [batch, num_heads, 1, 1],
                    'label_to_text': [batch, num_heads, 1, 1],
                    'semantic_to_struct': [batch, num_heads, 1, 1],
                    'struct_to_semantic': [batch, num_heads, 1, 1]
                }

        Returns:
            balance_loss: 模态平衡损失（标量）
        """
        # ========== 提取各模态的注意力权重（增强版） ==========

        # 获取device（用于创建tensor）
        device = next(iter(attention_weights_dict.values())).device if attention_weights_dict else torch.device('cpu')

        # Text的贡献度：从text_to_label中提取
        if 'text_to_label' in attention_weights_dict and attention_weights_dict['text_to_label'] is not None:
            text_attn = attention_weights_dict['text_to_label']
            w_text = text_attn.mean()
            # 源头预防: 检查数值异常
            if torch.isnan(w_text) or torch.isinf(w_text):
                print("  ⚠️ text_to_label权重异常，使用默认值")
                w_text = torch.tensor(0.33, device=device)
        else:
            w_text = torch.tensor(0.33, device=device)  # 默认均匀分布

        # Label的贡献度：从label_to_text中提取
        if 'label_to_text' in attention_weights_dict and attention_weights_dict['label_to_text'] is not None:
            label_attn = attention_weights_dict['label_to_text']
            w_label = label_attn.mean()
            if torch.isnan(w_label) or torch.isinf(w_label):
                print("  ⚠️ label_to_text权重异常，使用默认值")
                w_label = torch.tensor(0.33, device=device)
        else:
            w_label = torch.tensor(0.33, device=device)

        # Struct的贡献度：从struct_to_semantic中提取
        if 'struct_to_semantic' in attention_weights_dict and attention_weights_dict['struct_to_semantic'] is not None:
            struct_attn = attention_weights_dict['struct_to_semantic']
            w_struct = struct_attn.mean()
            if torch.isnan(w_struct) or torch.isinf(w_struct):
                print("  ⚠️ struct_to_semantic权重异常，使用默认值")
                w_struct = torch.tensor(0.33, device=device)
        else:
            w_struct = torch.tensor(0.33, device=device)

        # ========== 归一化权重（变成概率分布）- 增强版 ==========
        weights = torch.stack([w_text, w_label, w_struct])  # [3]

        # 源头预防: 检查权重是否异常
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print("  ⚠️ 检测到NaN或Inf，使用均匀分布")
            weights = torch.ones(3, device=weights.device) / 3
        else:
            # 确保非负
            weights = torch.abs(weights)

            # 确保权重不会太小（避免数值问题）
            weights = weights + 1e-8

            # Softmax归一化（确保和为1）
            weights = F.softmax(weights, dim=0)

        # ========== 方法1：熵最大化（鼓励均匀分布）- 增强版 ==========
        # 熵公式：H = -Σ p_i * log(p_i)
        # 理想情况：p_text = p_label = p_struct = 1/3，此时熵最大

        entropy = -torch.sum(weights * torch.log(weights + self.epsilon))

        # 源头预防: 检查熵是否异常
        if torch.isnan(entropy) or torch.isinf(entropy):
            print("  ⚠️ 熵计算异常，使用默认值")
            max_entropy = torch.log(torch.tensor(3.0, device=weights.device))
            entropy = max_entropy  # 使用最大熵

        # 归一化熵（最大熵为log(3)=1.0986）
        max_entropy = torch.log(torch.tensor(3.0, device=weights.device))
        normalized_entropy = entropy / (max_entropy + self.epsilon)

        # 损失 = -熵（因为要最大化熵，所以加负号）
        balance_loss_entropy = -entropy

        # ========== 方法2：MSE约束（强制接近均匀分布）==========
        # 目标：每个模态贡献1/3
        target = torch.tensor(1.0 / 3.0, device=weights.device)
        balance_loss_mse = ((weights - target) ** 2).sum()

        # ========== 组合两种方法 ==========
        # 熵最大化（主要）+ MSE约束（辅助）
        balance_loss = balance_loss_entropy + 0.5 * balance_loss_mse

        # ========== 可选：添加正则项（防止权重过小）==========
        # 确保每个模态至少有最小贡献（例如10%）
        min_weight = 0.1
        penalty = torch.relu(min_weight - weights).sum()
        balance_loss = balance_loss + 0.1 * penalty

        return self.weight * balance_loss