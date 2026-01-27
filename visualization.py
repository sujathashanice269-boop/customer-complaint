"""
Visualization Tools - visualization.py
For visualizing model attention weights, feature importance, etc.
Complete English version with all functions
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import networkx as nx
import os
import warnings
# ===== 复制开始 =====
from functools import lru_cache
from transformers import BertTokenizer
import jieba

# ============================================================
# 翻译模块 - 离线翻译器
# ============================================================

# ============================================================
# 翻译模块 - 手动字典优先 + MarianMT自动翻译兜底
# ============================================================

class SmartTranslator:
    """智能翻译器 - 手动字典优先，字典外使用MarianMT自动翻译"""

    BUILTIN_DICT = {
        # 基础网络相关
        "网络": "Network", "信号": "Signal", "网速": "Speed",
        "断网": "Offline", "掉线": "Drop", "卡顿": "Lag",
        "延迟": "Delay", "超时": "Timeout", "连接": "Connect",
        "上网": "Internet", "宽带": "Broadband", "WiFi": "WiFi",
        "4G": "4G", "5G": "5G", "基站": "Station", "覆盖": "Coverage",
        "网络问题": "NetworkIssue", "信号差": "WeakSignal",
        "网络故障": "NetworkFault", "网络慢": "SlowNetwork",
        # 费用相关
        "退费": "Refund", "退款": "Refund", "扣费": "Charge",
        "费用": "Fee", "话费": "Bill", "流量": "Data",
        "套餐": "Plan", "资费": "Tariff", "账单": "Bill",
        "余额": "Balance", "欠费": "Arrears", "充值": "Topup",
        "计费": "Billing", "多扣": "Overcharge", "乱扣": "WrongCharge",
        "资费争议": "BillingDispute", "套餐问题": "PlanIssue",
        "账户问题": "AccountIssue", "退费申请": "RefundRequest",
        # 服务相关
        "服务": "Service", "客服": "Support", "投诉": "Complaint",
        "处理": "Handle", "解决": "Resolve", "回复": "Reply",
        "响应": "Response", "等待": "Wait", "态度": "Attitude",
        "服务质量": "ServiceQuality", "响应速度": "ResponseSpeed",
        "处理超时": "HandleTimeout", "服务态度": "ServiceAttitude",
        # 问题状态
        "问题": "Issue", "故障": "Fault", "错误": "Error",
        "异常": "Abnormal", "无法": "Cannot", "不能": "Cannot",
        "失败": "Fail", "慢": "Slow", "差": "Poor", "弱": "Weak",
        # 操作动作
        "反映": "Report", "咨询": "Consult", "查询": "Query",
        "办理": "Apply", "申请": "Request", "取消": "Cancel",
        "变更": "Change", "要求": "Demand", "需要": "Need",
        "业务办理": "Business", "业务": "Business",
        # 用户设备
        "用户": "User", "客户": "Customer", "手机": "Phone",
        "电话": "Call", "短信": "SMS", "号码": "Number", "设备": "Device",
        "移动": "Mobile", "联通": "Unicom", "电信": "Telecom",
        # 业务类型
        "宽带故障": "BroadbandFault", "移动网络": "MobileNetwork",
        "固网": "FixedNetwork", "增值业务": "VAS", "国际漫游": "Roaming",
        "携号转网": "PortNumber", "实名认证": "RealName",
        "合约": "Contract", "违约金": "Penalty", "押金": "Deposit",
        "发票": "Invoice", "积分": "Points", "优惠": "Discount",
        "促销": "Promotion", "活动": "Campaign",
        # 操作状态
        "使用": "Use", "开通": "Activate", "关闭": "Close",
        "升级": "Upgrade", "降级": "Downgrade", "续费": "Renew",
        "缴费": "Payment", "欠费停机": "Suspended",
        "原因": "Reason", "结果": "Result", "建议": "Suggest",
        "满意": "Satisfied", "不满": "Unsatisfied", "重复": "Repeat",
        # 新增：标签层级常用词
        "一级": "Level1", "二级": "Level2", "三级": "Level3", "四级": "Level4",
        "类别": "Category", "分类": "Class", "类型": "Type",
        "主题": "Topic", "原因分析": "RootCause", "处理结果": "Resolution",
        "紧急": "Urgent", "一般": "Normal", "重要": "Important",
        # 新增：投诉文本常用词
        "不好": "Bad", "太慢": "TooSlow", "不行": "NotWork",
        "打不通": "CannotCall", "上不了": "CannotAccess", "收不到": "CannotReceive",
        "发不出": "CannotSend", "看不了": "CannotView", "用不了": "CannotUse",
        "经常": "Often", "一直": "Always", "有时": "Sometimes",
        "突然": "Suddenly", "总是": "Always", "从不": "Never",
        "已经": "Already", "还是": "Still", "仍然": "Still",
        "希望": "Hope", "麻烦": "Trouble",
        "感谢": "Thanks", "谢谢": "Thanks", "抱歉": "Sorry",
        # 新增：数字和单位
        "天": "Days", "小时": "Hours", "分钟": "Minutes",
        "元": "Yuan", "块": "Yuan", "钱": "Money",
        "次": "Times", "个": "Count", "条": "Items",
        # 新增：更多业务术语
        "月租": "MonthlyFee", "包月": "Monthly", "日租": "DailyFee",
        "本地": "Local", "长途": "LongDistance", "漫游费": "RoamingFee",
        "国内": "Domestic", "国际": "International", "港澳台": "HMT",
        "语音": "Voice", "视频": "Video",
        "彩信": "MMS", "来电显示": "CallerID", "呼叫转移": "CallForward",
        "留言": "Voicemail", "黑名单": "Blacklist", "白名单": "Whitelist",
    }

    def __init__(self):
        self.translator_model = None
        self.translator_tokenizer = None
        self.auto_translate_cache = {}
        self.model_loaded = False
        self._try_init_translator()

    def _try_init_translator(self):
        try:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = 'Helsinki-NLP/opus-mt-zh-en'
            print(f"🔄 正在加载自动翻译模型: {model_name}")
            self.translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translator_model = MarianMTModel.from_pretrained(model_name)
            self.model_loaded = True
            print("✅ 自动翻译模型加载成功")
            print(f"✅ 内置字典词汇数: {len(self.BUILTIN_DICT)}")
        except Exception as e:
            print(f"⚠️ 自动翻译模型加载失败: {e}")
            print(f"✅ 将仅使用内置字典翻译 (词汇数: {len(self.BUILTIN_DICT)})")
            self.model_loaded = False

    def _auto_translate(self, text: str) -> str:
        if not self.model_loaded:
            return text
        if text in self.auto_translate_cache:
            return self.auto_translate_cache[text]
        try:
            inputs = self.translator_tokenizer(text, return_tensors="pt", padding=True)
            translated = self.translator_model.generate(**inputs)
            result = self.translator_tokenizer.decode(translated[0], skip_special_tokens=True).strip()
            if ' ' in result and len(text) <= 4:
                result = result.split()[0]
            self.auto_translate_cache[text] = result
            return result
        except:
            return text

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        text = text.strip().replace("##", "")
        if text in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
            return ""
        if text.replace(' ', '').isascii():
            return text
        if text in self.BUILTIN_DICT:
            return self.BUILTIN_DICT[text]
        return self._auto_translate(text)

    def translate_batch(self, texts: List[str]) -> List[str]:
        results = []
        to_auto_translate = []
        to_auto_translate_indices = []
        for i, text in enumerate(texts):
            text = text.strip().replace("##", "") if text else ""
            if not text or text in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                results.append("")
            elif text.replace(' ', '').isascii():
                results.append(text)
            elif text in self.BUILTIN_DICT:
                results.append(self.BUILTIN_DICT[text])
            elif text in self.auto_translate_cache:
                results.append(self.auto_translate_cache[text])
            else:
                results.append(None)
                to_auto_translate.append(text)
                to_auto_translate_indices.append(i)
        if to_auto_translate and self.model_loaded:
            try:
                inputs = self.translator_tokenizer(to_auto_translate, return_tensors="pt", padding=True)
                translated = self.translator_model.generate(**inputs)
                for j, idx in enumerate(to_auto_translate_indices):
                    result = self.translator_tokenizer.decode(translated[j], skip_special_tokens=True).strip()
                    original = to_auto_translate[j]
                    if ' ' in result and len(original) <= 4:
                        result = result.split()[0]
                    self.auto_translate_cache[original] = result
                    results[idx] = result
            except:
                for j, idx in enumerate(to_auto_translate_indices):
                    results[idx] = to_auto_translate[j]
        else:
            for j, idx in enumerate(to_auto_translate_indices):
                results[idx] = to_auto_translate[j]
        return results


_global_translator = None


def get_translator():
    global _global_translator
    if _global_translator is None:
        _global_translator = SmartTranslator()
    return _global_translator


def translate_label_path(label_path: str) -> list:
    if '→' in label_path:
        parts = label_path.split('→')
    elif '->' in label_path:
        parts = label_path.split('->')
    else:
        parts = [label_path]
    translator = get_translator()
    clean_parts = [p.strip() for p in parts if p.strip()]
    return translator.translate_batch(clean_parts)


# ============================================================
# Jieba分词支持 - 用于词级别attention聚合
# ============================================================

_jieba_initialized = False


def init_jieba_with_user_dict(user_dict_file='new_user_dict.txt'):
    """初始化jieba并加载用户词典（仅用于可视化）"""
    global _jieba_initialized
    if not _jieba_initialized:
        if os.path.exists(user_dict_file):
            jieba.load_userdict(user_dict_file)
            print(f"✅ Visualization: 已加载用户词典 {user_dict_file}")
        else:
            print(f"⚠️ Visualization: 用户词典不存在 {user_dict_file}")
        _jieba_initialized = True


def aggregate_tokens_to_words(tokens, attention_scores, text, user_dict_file='new_user_dict.txt'):
    """
    将BERT字级别的tokens和attention聚合为jieba词级别

    Args:
        tokens: BERT分词后的token列表 ['[CLS]', '网', '络', '信', '号', ...]
        attention_scores: 每个token的attention分数 (1D array)
        text: 原始中文文本
        user_dict_file: 用户词典文件路径

    Returns:
        word_list: jieba分词后的词语列表
        word_attention: 每个词语的聚合attention分数
        word_token_indices: 每个词语对应的原始token索引范围 [(start, end), ...]
    """
    # 初始化jieba
    init_jieba_with_user_dict(user_dict_file)

    # 用jieba对原文分词
    jieba_words = list(jieba.cut(text))

    # 过滤掉空白词
    jieba_words = [w for w in jieba_words if w.strip()]

    # 清理tokens：去掉[CLS]、[SEP]等特殊token，建立位置映射
    # token_map: [(原始索引, 清理后的字符), ...]
    special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]', ''}
    token_map = []
    for i, token in enumerate(tokens):
        clean_token = token.replace('##', '')
        if token not in special_tokens and clean_token.strip():
            token_map.append((i, clean_token))

    # 将jieba词语与BERT tokens对齐
    word_list = []
    word_attention = []
    word_token_indices = []

    token_ptr = 0  # 当前在token_map中的位置

    for word in jieba_words:
        word_chars = list(word)
        word_len = len(word_chars)

        # 查找这个词在token_map中的起始位置
        start_ptr = token_ptr
        matched_indices = []
        matched_attentions = []

        # 尝试匹配这个词的所有字符
        temp_ptr = token_ptr
        match_success = True

        for char in word_chars:
            if temp_ptr >= len(token_map):
                match_success = False
                break

            orig_idx, token_char = token_map[temp_ptr]

            # 检查字符是否匹配（处理可能的空格和特殊字符）
            if token_char == char or char in token_char:
                matched_indices.append(orig_idx)
                matched_attentions.append(attention_scores[orig_idx] if orig_idx < len(attention_scores) else 0)
                temp_ptr += 1
            else:
                # 字符不匹配，可能是特殊字符，跳过
                match_success = False
                break

        if match_success and matched_indices:
            # 成功匹配，聚合attention（使用平均值）
            word_list.append(word)
            word_attention.append(np.mean(matched_attentions))
            word_token_indices.append((matched_indices[0], matched_indices[-1]))
            token_ptr = temp_ptr
        else:
            # 匹配失败，尝试跳过当前token继续
            if token_ptr < len(token_map):
                token_ptr += 1

    return word_list, np.array(word_attention), word_token_indices

def select_top_attention_tokens(attention_matrix, tokens, top_k=8, text=None, user_dict_file='new_user_dict.txt'):
    """
    选择注意力权重最高的tokens（支持词级别聚合）

    改进：当提供text参数时，使用jieba分词进行词级别聚合
         否则回退到原来的字级别选择

    Args:
        attention_matrix: attention权重矩阵 [seq_len, key_len] 或 [seq_len]
        tokens: BERT分词后的token列表
        top_k: 选择top-k个词/字
        text: 原始文本（可选，提供时启用词级别聚合）
        user_dict_file: 用户词典文件路径

    Returns:
        final_indices: 选中的token索引列表
        final_orig: 原始词语/字符列表
        final_trans: 翻译后的词语列表
    """
    # 确保attention_matrix是2D的
    if attention_matrix.ndim == 1:
        attention_matrix = attention_matrix.reshape(-1, 1)

    # 计算每个token的importance（对所有key位置求平均）
    importance = attention_matrix.mean(axis=1)

    # 需要过滤的特殊符号和无意义字符
    special_chars = {'#', '##', '@', '$', '%', '^', '&', '*', '(', ')',
                     '-', '_', '+', '=', '[', ']', '{', '}', '|', '\\',
                     '/', '<', '>', ',', '.', '?', '!', '~', '`', '"', "'",
                     '：', '；', '，', '。', '！', '？', '、', '"', '"', ''', ''',
                     '（', '）', '【', '】', '《', '》', '…', '—', '·'}

    # ========== 新增：词级别聚合模式 ==========
    if text is not None and len(text.strip()) > 0:
        try:
            # 使用jieba分词进行词级别聚合
            word_list, word_attention, word_token_indices = aggregate_tokens_to_words(
                tokens, importance, text, user_dict_file
            )

            if len(word_list) > 0:
                # 过滤无意义的词
                valid_words = []
                for i, (word, attn, indices) in enumerate(zip(word_list, word_attention, word_token_indices)):
                    # 过滤单字符且为特殊符号的
                    if len(word) == 1 and word in special_chars:
                        continue
                    # 过滤纯特殊符号组成的词
                    if all(c in special_chars or c.isspace() for c in word):
                        continue
                    # 过滤attention过低的
                    if attn <= 0.001:
                        continue
                    valid_words.append((i, word, attn, indices))

                # 按attention排序，选择top-k
                valid_words.sort(key=lambda x: x[2], reverse=True)
                selected = valid_words[:top_k]

                # 按原始顺序排序
                selected.sort(key=lambda x: x[3][0])

                # 翻译
                translator = get_translator()
                final_indices = [s[3][0] for s in selected]  # 使用词的起始token索引
                final_orig = [s[1] for s in selected]
                final_trans = translator.translate_batch(final_orig)

                # 过滤翻译后为空的结果
                result_indices = []
                result_orig = []
                result_trans = []
                for idx, orig, trans in zip(final_indices, final_orig, final_trans):
                    if trans and trans.strip() and trans not in special_chars:
                        result_indices.append(idx)
                        result_orig.append(orig)
                        result_trans.append(trans)

                if result_trans:
                    print(f"   [词级别聚合] 提取到 {len(result_trans)} 个关键词")
                    return result_indices, result_orig, result_trans

        except Exception as e:
            print(f"   ⚠️ 词级别聚合失败，回退到字级别: {e}")

    # ========== 原有逻辑：字级别选择（作为fallback） ==========
    valid = []

    for i, (token, score) in enumerate(zip(tokens, importance)):
        # 过滤BERT特殊token
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '', '[MASK]']:
            continue
        # 过滤低权重token
        if score <= 0.001:
            continue
        # 去掉##前缀
        clean_token = token.replace('##', '')
        # 过滤空字符串
        if not clean_token or clean_token.strip() == '':
            continue
        # 过滤纯特殊符号
        if clean_token in special_chars:
            continue
        # 过滤只包含特殊符号的token
        if all(c in special_chars or c.isspace() for c in clean_token):
            continue
        valid.append((i, token, score))

    valid.sort(key=lambda x: x[2], reverse=True)
    selected = valid[:top_k]
    selected.sort(key=lambda x: x[0])

    translator = get_translator()
    indices = [s[0] for s in selected]
    orig_tokens = [s[1].replace('##', '') for s in selected]
    trans_tokens = translator.translate_batch(orig_tokens)

    # 过滤翻译后为空或仍是特殊符号的结果
    final_indices = []
    final_orig = []
    final_trans = []
    for idx, orig, trans in zip(indices, orig_tokens, trans_tokens):
        # 跳过翻译后为空的
        if not trans or trans.strip() == '':
            continue
        # 跳过翻译后仍是特殊符号的
        if trans in special_chars or all(c in special_chars or c.isspace() for c in trans):
            continue
        final_indices.append(idx)
        final_orig.append(orig)
        final_trans.append(trans)

    return final_indices, final_orig, final_trans


warnings.filterwarnings('ignore')

# ============================================================
# Matplotlib 中文字体配置 - SimHei
# ============================================================

# 方法1: 尝试直接使用字体名称
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 方法2: 如果方法1不行，直接指定字体文件路径
import os
simhei_paths = [
    '/usr/share/fonts/chinese/simhei.ttf',
    '/usr/share/fonts/truetype/simhei.ttf',
    '/usr/local/share/fonts/simhei.ttf',
    os.path.expanduser('~/.fonts/simhei.ttf'),
]

simhei_path = None
for path in simhei_paths:
    if os.path.exists(path):
        simhei_path = path
        break

if simhei_path:
    # 注册字体
    fm.fontManager.addfont(simhei_path)
    prop = fm.FontProperties(fname=simhei_path)
    plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
    print(f"✅ SimHei font loaded from: {simhei_path}")
else:
    print("⚠️ SimHei font not found, using fallback fonts")

# 其他配置
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 验证字体是否可用
print("=" * 50)
print("Font Configuration Check:")
available_fonts = set([f.name for f in fm.fontManager.ttflist])
chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
for font in chinese_fonts:
    if font in available_fonts:
        print(f"  ✅ {font} - Available")
    else:
        print(f"  ❌ {font} - Not Found")
print("=" * 50)


class AttentionVisualizer:
    """Attention Weight Visualizer"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def visualize_cross_modal_attention(self, attention_weights: Dict[str, torch.Tensor],
                                      sample_text: str = None, save_path: str = None):
        """Visualize cross-modal attention weights"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        attention_types = [
            ('text_to_label', 'Text → Label'),
            ('text_to_struct', 'Text → Struct'),
            ('label_to_text', 'Label → Text'),
            ('label_to_struct', 'Label → Struct'),
            ('struct_to_text', 'Struct → Text'),
            ('struct_to_label', 'Struct → Label')
        ]

        for idx, (attn_key, title) in enumerate(attention_types):
            if attn_key in attention_weights:
                attn = attention_weights[attn_key][0, 0].cpu().numpy()
                ax = axes[idx]
                sns.heatmap(attn, cmap='Blues', ax=ax, cbar=True,
                           xticklabels=False, yticklabels=False)
                ax.set_title(title, fontsize=14)

                if idx == 0 and sample_text:
                    ax.text(0.5, -0.15, f"Text: {sample_text[:30]}...",
                           transform=ax.transAxes, ha='center', fontsize=10)

        plt.suptitle('Cross-Modal Attention Weights Visualization', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_token_attention(self, text: str, attention_scores: np.ndarray,
                                save_path: str = None):
        """Visualize token-level attention scores"""
        tokens = list(text)[:50]
        fig, ax = plt.subplots(figsize=(15, 3))

        if len(attention_scores) > len(tokens):
            attention_scores = attention_scores[:len(tokens)]

        norm = plt.Normalize(vmin=attention_scores.min(), vmax=attention_scores.max())
        colors = plt.cm.Blues(norm(attention_scores))

        for i, (token, color) in enumerate(zip(tokens, colors)):
            ax.text(i, 0, token, ha='center', va='center',
                   color='black', backgroundcolor=color, fontsize=12)

        ax.set_xlim(-1, len(tokens))
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.set_title('Token-Level Attention Visualization', fontsize=14)

        sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Attention Intensity')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class FeatureImportanceVisualizer:
    """Feature Importance Visualizer"""

    def __init__(self):
        pass

    def visualize_structured_features(self, feature_names: List[str],
                                    feature_importance: np.ndarray,
                                    save_path: str = None):
        """Visualize structured feature importance"""
        indices = np.argsort(feature_importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = feature_importance[indices]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(sorted_features, sorted_importance)

        colors = plt.cm.RdYlBu_r(sorted_importance / sorted_importance.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('Feature Importance')
        plt.title('Structured Feature Importance Ranking')
        plt.grid(axis='x', alpha=0.3)

        for i, v in enumerate(sorted_importance):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class LabelTreeVisualizer:
    """Label Tree Visualizer"""

    def __init__(self):
        pass

    def visualize_label_tree(self, label_tree, save_path: str = None):
        """Visualize complaint label hierarchy"""
        G = nx.DiGraph()

        for node_id, label in label_tree.id_to_label.items():
            G.add_node(node_id, label=label)

        for parent, child in label_tree.edges:
            G.add_edge(parent, child)

        pos = nx.spring_layout(G, k=2, iterations=50)

        plt.figure(figsize=(15, 10))

        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                              arrowsize=20, alpha=0.6)

        node_colors = []
        for node in G.nodes():
            level = len(nx.ancestors(G, node))
            node_colors.append(level)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              cmap='viridis', node_size=1000, alpha=0.8)

        labels = nx.get_node_attributes(G, 'label')
        short_labels = {k: v[:10] + '...' if len(v) > 10 else v
                       for k, v in labels.items()}
        nx.draw_networkx_labels(G, pos, short_labels, font_size=8)

        plt.title('Complaint Label Hierarchy Structure', fontsize=16)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class PredictionAnalyzer:
    """Prediction Result Analyzer"""

    def __init__(self):
        pass

    def visualize_prediction_distribution(self, predictions: pd.DataFrame,
                                        save_path: str = None):
        """Visualize prediction result distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Prediction class distribution
        ax = axes[0, 0]
        predictions['prediction_text'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Prediction Class Distribution')
        ax.set_xlabel('Prediction Class')
        ax.set_ylabel('Count')

        # 2. Confidence distribution
        ax = axes[0, 1]
        ax.hist(predictions['confidence'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title('Prediction Confidence Distribution')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.axvline(0.8, color='red', linestyle='--', label='High Confidence Threshold')
        ax.legend()

        # 3. Repeat complaint probability distribution
        ax = axes[1, 0]
        ax.hist(predictions['repeat_probability'], bins=20, alpha=0.7,
               color='red', edgecolor='black')
        ax.set_title('Repeat Complaint Probability Distribution')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Frequency')
        ax.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
        ax.legend()

        # 4. Empty subplot for future use
        axes[1, 1].axis('off')

        plt.suptitle('Prediction Results Analysis', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class TextAnalyzer:
    """Text Analysis and Visualization"""

    def __init__(self):
        pass

    def visualize_wordcloud(self, texts_by_category: Dict[str, List[str]],
                           save_path: str = None):
        """Visualize word clouds by category"""
        categories = list(texts_by_category.keys())
        n_categories = len(categories)

        fig, axes = plt.subplots(1, n_categories, figsize=(6*n_categories, 5))
        if n_categories == 1:
            axes = [axes]

        for idx, (category, texts) in enumerate(texts_by_category.items()):
            if texts:
                combined_text = ' '.join(texts)

                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(combined_text)

                ax = axes[idx]
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'{category} Word Cloud', fontsize=16)
                ax.axis('off')
            else:
                axes[idx].text(0.5, 0.5, f'No {category} Data',
                             ha='center', va='center', fontsize=14)
                axes[idx].axis('off')

        plt.suptitle('Complaint Text Word Cloud Analysis', fontsize=18)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def visualize_training_curves(history: Dict[str, List[float]], save_path: str = None):
    """Visualize training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('f1', 'F1 Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('auc', 'AUC')
    ]

    for idx, (metric, title) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        if f'train_{metric}' in history:
            ax.plot(history[f'train_{metric}'], label='Train', marker='o')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label='Validation', marker='s')

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Progress Metrics', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# Enhanced Visualization Classes for Paper-Quality Figures
# ============================================================

class EnhancedAttentionVisualizer:
    """
    Enhanced Attention Visualizer - For paper-quality figure generation
    Supports cross-modal attention heatmaps, case decision tracing, etc.
    """

    def __init__(self, tokenizer=None, save_dir: str = './outputs/figures'):
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_cross_modal_attention_heatmap(self,
                                           attention_weights: Dict[str, torch.Tensor],
                                           struct_features: List[str] = None,
                                           struct_feature_names: List[str] = None,
                                           sample_id: str = "sample",
                                           label_path: str = None,
                                           tokenizer=None,
                                           text: str = None,
                                           new_code: str = None,
                                           save_path: str = None) -> plt.Figure:
        """
        Plot cross-modal attention heatmap
        修复: 添加Y轴文本关键词显示
        """
        if new_code:
            print(f"📌 Processing sample new_code: {new_code}")

        attn_types = []
        titles = []
        if 'text_to_label' in attention_weights and attention_weights['text_to_label'] is not None:
            attn_types.append(('text_to_label', attention_weights['text_to_label']))
            titles.append('Text -> Label')
        if 'label_to_text' in attention_weights and attention_weights['label_to_text'] is not None:
            attn_types.append(('label_to_text', attention_weights['label_to_text']))
            titles.append('Label -> Text')
        if 'text_to_struct' in attention_weights and attention_weights['text_to_struct'] is not None:
            attn_types.append(('text_to_struct', attention_weights['text_to_struct']))
            titles.append('Text -> Structured')

        if len(attn_types) == 0:
            print("Warning: No available attention weights")
            return None

        # 翻译标签路径
        translated_labels = None
        if label_path:
            translated_labels = translate_label_path(label_path)
            print(f"   Label: {' -> '.join(translated_labels)}")

        # 【修复问题2】提取文本关键词用于Y轴显示
        text_keywords = None
        text_keyword_indices = None
        if tokenizer is not None and text is not None:
            try:
                for attn_name, attn_tensor in attn_types:
                    if 'text_to' in attn_name:
                        if isinstance(attn_tensor, torch.Tensor):
                            attn_for_keywords = attn_tensor.detach().cpu()
                            if attn_for_keywords.dim() == 4:
                                attn_for_keywords = attn_for_keywords[0].mean(dim=0).numpy()
                            elif attn_for_keywords.dim() == 3:
                                attn_for_keywords = attn_for_keywords[0].numpy()
                            else:
                                attn_for_keywords = attn_for_keywords.numpy()
                        else:
                            attn_for_keywords = np.array(attn_tensor)

                        if attn_for_keywords.ndim == 1:
                            attn_for_keywords = attn_for_keywords.reshape(1, -1)

                        encoding = tokenizer(text, max_length=256, truncation=True, return_tensors='pt')
                        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

                        text_keyword_indices, orig_tokens, text_keywords = select_top_attention_tokens(
                            attn_for_keywords, tokens, top_k=min(8, attn_for_keywords.shape[0]), text=text
                        )

                        if text_keywords:
                            print(f"   Text Keywords (Y-axis): {', '.join(text_keywords)}")
                        break
            except Exception as e:
                print(f"   Warning: Failed to extract text keywords: {e}")
                text_keywords = None

        n_plots = len(attn_types)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, ((attn_name, attn_tensor), title) in enumerate(zip(attn_types, titles)):
            ax = axes[idx]
            if isinstance(attn_tensor, torch.Tensor):
                attn = attn_tensor.detach().cpu()
                if attn.dim() == 4:
                    attn = attn[0].mean(dim=0).numpy()
                elif attn.dim() == 3:
                    attn = attn[0].numpy()
                elif attn.dim() == 2:
                    attn = attn.numpy()
                else:
                    attn = attn.numpy()
            else:
                attn = np.array(attn_tensor)

            if attn.ndim == 1:
                attn = attn.reshape(1, -1)

            # 【修复问题2】如果有文本关键词，对attention矩阵进行行选择
            y_labels = None
            if 'text_to' in attn_name and text_keywords and text_keyword_indices:
                valid_indices = [i for i in text_keyword_indices if i < attn.shape[0]]
                if valid_indices:
                    attn = attn[valid_indices, :]
                    y_labels = text_keywords[:len(valid_indices)]

            im = ax.imshow(attn, cmap='YlOrRd', aspect='auto')
            ax.set_title(title, fontsize=14, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Attention Weight', fontsize=10)

            # 设置X轴标签
            n_cols_attn = attn.shape[1]
            if 'label' in attn_name and translated_labels:
                if len(translated_labels) >= n_cols_attn:
                    x_labels = translated_labels[:n_cols_attn]
                else:
                    x_labels = translated_labels + [f"L{i + 1}" for i in range(len(translated_labels), n_cols_attn)]
                ax.set_xticks(range(n_cols_attn))
                ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=9)
                ax.set_xlabel('Label Hierarchy', fontsize=11)
            elif 'struct' in attn_name and struct_feature_names:
                if len(struct_feature_names) >= n_cols_attn:
                    x_labels = struct_feature_names[:n_cols_attn]
                else:
                    x_labels = struct_feature_names + [f"F{i + 1}" for i in
                                                       range(len(struct_feature_names), n_cols_attn)]
                ax.set_xticks(range(n_cols_attn))
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
                ax.set_xlabel('Structured Features', fontsize=11)
            else:
                ax.set_xlabel('Key Position', fontsize=11)

            # 【修复问题2】设置Y轴标签 - 显示文本关键词
            if y_labels and 'text_to' in attn_name:
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels(y_labels, fontsize=9)
                ax.set_ylabel('Text Keywords', fontsize=11)
            else:
                ax.set_ylabel('Query Position', fontsize=11)

        for idx in range(len(attn_types), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Cross-Modal Attention (Sample: {sample_id})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, f'attention_heatmap_{sample_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Attention heatmap saved: {save_path}")

        return fig

    def plot_attention_with_text(self,
                                 attention_weights: Dict[str, torch.Tensor],
                                 text: str,
                                 label_path: str,
                                 prediction: int,
                                 confidence: float,
                                 true_label: int = None,
                                 sample_id: str = "case",
                                 new_code: str = None,
                                 tokenizer=None,
                                 struct_feature_names: List[str] = None,
                                 save_path: str = None) -> plt.Figure:
        """Plot attention heatmap with text and label - 带翻译和关键词挑选"""
        plt.rcParams['font.family'] = 'DejaVu Sans'

        if new_code:
            print(f"📌 Case study sample new_code: {new_code}")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 2.2, 1], hspace=0.3, wspace=0.3)

        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis('off')

        # 【修复问题3】显示文本摘要（前100个字符）
        display_id = new_code if new_code else sample_id
        text_preview = text[:100] + "..." if len(text) > 100 else text
        translator = get_translator()
        translated_preview = translator.translate(text_preview)
        info_text = f"Sample: {display_id}\nText Preview: {translated_preview}"
        ax_text.text(0.5, 0.5, info_text, ha='center', va='center',
                     fontsize=10, wrap=True,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

        translated_labels = translate_label_path(label_path)
        n_labels = len(translated_labels)
        print(f"   Label Path: {' → '.join(translated_labels)}")

        ax_label = fig.add_subplot(gs[2, 0])
        ax_label.axis('off')
        label_display = " → ".join(translated_labels)
        ax_label.text(0.5, 0.5, f"Label Path:\n{label_display}", ha='center', va='center',
                      fontsize=10, wrap=True,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

        ax_pred = fig.add_subplot(gs[2, 1])
        ax_pred.axis('off')
        pred_text = "Repeat Complaint" if prediction == 1 else "Non-Repeat"
        true_text = ""
        if true_label is not None:
            true_text = f"\nTrue: {'Repeat' if true_label == 1 else 'Non-Repeat'}"
            is_correct = prediction == true_label
            status = " (Correct)" if is_correct else " (Wrong)"
            true_text += status

        pred_content = f"Prediction: {pred_text}\nConfidence: {confidence:.2%}{true_text}"
        color = 'lightcoral' if prediction == 1 else 'lightgreen'
        ax_pred.text(0.5, 0.5, pred_content, ha='center', va='center',
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.5))

        ax_attn1 = fig.add_subplot(gs[1, 0])
        ax_attn2 = fig.add_subplot(gs[1, 1])

        if 'text_to_label' in attention_weights and attention_weights['text_to_label'] is not None:
            attn = attention_weights['text_to_label']
            if isinstance(attn, torch.Tensor):
                attn = attn.detach().cpu()
                if attn.dim() >= 3:
                    attn = attn[0].mean(dim=0).numpy() if attn.dim() == 4 else attn[0].numpy()
                else:
                    attn = attn.numpy()
            if attn.ndim == 1:
                attn = attn.reshape(1, -1)

            y_labels = None
            top_k = 10
            if tokenizer is not None:
                try:
                    encoding = tokenizer(text, max_length=256, truncation=True, return_tensors='pt')
                    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
                    indices, orig_tokens, trans_tokens = select_top_attention_tokens(attn, tokens, top_k=top_k, text=text)

                    print(f"   Top-{len(indices)} keywords:")
                    for orig, trans in zip(orig_tokens, trans_tokens):
                        print(f"      {orig} → {trans}")

                    attn = attn[indices, :]
                    if attn.shape[1] > n_labels:
                        attn = attn[:, :n_labels]
                    y_labels = trans_tokens
                except Exception as e:
                    print(f"   ⚠️ Token selection error: {e}")

            im1 = ax_attn1.imshow(attn, cmap='Blues', aspect='auto')
            ax_attn1.set_title('Text → Label Attention', fontsize=12, fontweight='bold')

            n_cols = attn.shape[1]
            if len(translated_labels) >= n_cols:
                x_labels = translated_labels[:n_cols]
            else:
                x_labels = translated_labels + [f"L{i + 1}" for i in range(len(translated_labels), n_cols)]
            ax_attn1.set_xticks(range(n_cols))
            ax_attn1.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=9)
            ax_attn1.set_xlabel('Label Hierarchy', fontsize=10)

            if y_labels:
                ax_attn1.set_yticks(range(len(y_labels)))
                ax_attn1.set_yticklabels(y_labels, fontsize=9)
                ax_attn1.set_ylabel('Text Keywords', fontsize=10)
            else:
                ax_attn1.set_ylabel('Text Position', fontsize=10)

            plt.colorbar(im1, ax=ax_attn1, shrink=0.6)
        else:
            ax_attn1.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            ax_attn1.set_title('Text → Label Attention', fontsize=12)

        if 'text_to_struct' in attention_weights and attention_weights['text_to_struct'] is not None:
            attn = attention_weights['text_to_struct']
            if isinstance(attn, torch.Tensor):
                attn = attn.detach().cpu()
                if attn.dim() >= 3:
                    attn = attn[0].mean(dim=0).numpy() if attn.dim() == 4 else attn[0].numpy()
                else:
                    attn = attn.numpy()
            if attn.ndim == 1:
                attn = attn.reshape(1, -1)

            # 【修复问题2】添加Y轴文本关键词
            y_labels_struct = None
            if tokenizer is not None:
                try:
                    encoding = tokenizer(text, max_length=256, truncation=True, return_tensors='pt')
                    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
                    indices_struct, orig_tokens_struct, trans_tokens_struct = select_top_attention_tokens(
                        attn, tokens, top_k=min(10, attn.shape[0]), text=text
                    )

                    if indices_struct and trans_tokens_struct:
                        # 对attention矩阵进行行选择
                        valid_indices = [i for i in indices_struct if i < attn.shape[0]]
                        if valid_indices:
                            attn = attn[valid_indices, :]
                            y_labels_struct = trans_tokens_struct[:len(valid_indices)]
                except Exception as e:
                    print(f"   ⚠️ Text→Struct keyword extraction error: {e}")

            im2 = ax_attn2.imshow(attn, cmap='Oranges', aspect='auto')
            # 【修复问题2】标题从"Semantic"改为"Text"
            ax_attn2.set_title('Text → Structured Attention', fontsize=12, fontweight='bold')

            # 【修复问题2】X轴显示结构化特征名称
            n_struct_cols = attn.shape[1]
            if struct_feature_names and len(struct_feature_names) > 0:
                if len(struct_feature_names) >= n_struct_cols:
                    x_labels_struct = struct_feature_names[:n_struct_cols]
                else:
                    x_labels_struct = struct_feature_names + [f"F{i + 1}" for i in
                                                              range(len(struct_feature_names), n_struct_cols)]
                ax_attn2.set_xticks(range(n_struct_cols))
                ax_attn2.set_xticklabels(x_labels_struct, rotation=45, ha='right', fontsize=8)
                ax_attn2.set_xlabel('Structured Features', fontsize=10)
            else:
                ax_attn2.set_xlabel('Feature Index', fontsize=10)

            # 【修复问题2】Y轴显示文本关键词
            if y_labels_struct:
                ax_attn2.set_yticks(range(len(y_labels_struct)))
                ax_attn2.set_yticklabels(y_labels_struct, fontsize=9)
                ax_attn2.set_ylabel('Text Keywords', fontsize=10)
            else:
                ax_attn2.set_ylabel('Text Position', fontsize=10)

            plt.colorbar(im2, ax=ax_attn2, shrink=0.6)
        else:
            ax_attn2.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            ax_attn2.set_title('Text → Structured Attention', fontsize=12)

        plt.suptitle(f'Case Study: {sample_id}', fontsize=16, fontweight='bold')

        if save_path is None:
            save_path = os.path.join(self.save_dir, f'case_study_{sample_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Case study figure saved: {save_path}")

        return fig


class ModalityContributionAnalyzer:
    """
    Modality Contribution Analyzer - Quantify each modality's contribution
    """

    def __init__(self, save_dir: str = './outputs/figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_ablation_comparison(self,
                                  ablation_results: Dict[str, Dict[str, float]],
                                  save_path: str = None) -> plt.Figure:
        """
        Plot ablation study comparison - Modality contribution bar chart
        """
        models = list(ablation_results.keys())
        metrics = ['accuracy', 'f1', 'auc']

        model_names = {
            'full_model': 'Full Model',
            'text_only': 'Text Only',
            'label_only': 'Label Only',
            'struct_only': 'Struct Only',
            'text_label': 'Text+Label',
            'text_struct': 'Text+Struct',
            'label_struct': 'Label+Struct',
            'No_pretrain': 'No_Pretrain'
        }

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(models))
        width = 0.25

        colors = ['#2ecc71', '#3498db', '#e74c3c']

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [ablation_results[m].get(metric, 0) for m in models]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=metric.upper(), color=color, alpha=0.85)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model Configuration', fontsize=12)
        ax.set_ylabel('Performance Score', fontsize=12)
        ax.set_title('Ablation Study Results Comparison', fontsize=14, fontweight='bold')

        ax.set_xticks(x)
        display_names = [model_names.get(m, m) for m in models]
        ax.set_xticklabels(display_names, rotation=30, ha='right', fontsize=10)

        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'ablation_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Ablation comparison figure saved: {save_path}")

        return fig

    def plot_modality_contribution_pie(self,
                                        contributions: Dict[str, float],
                                        save_path: str = None) -> plt.Figure:
        """
        Plot modality contribution pie chart
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        labels = list(contributions.keys())
        sizes = list(contributions.values())
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        explode = (0.05, 0.05, 0.05)

        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90,
                                           textprops={'fontsize': 12})

        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        ax.set_title('Modality Contribution to Prediction', fontsize=16, fontweight='bold')

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'modality_contribution_pie.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Modality contribution pie chart saved: {save_path}")

        return fig

    def plot_radar_comparison(self,
                               model_results: Dict[str, Dict[str, float]],
                               save_path: str = None) -> plt.Figure:
        """
        Plot model performance radar chart comparison
        """
        metrics = ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']
        metric_keys = ['auc', 'f1', 'precision', 'recall', 'accuracy']

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

        model_display_names = {
            'full_model': 'Full Model',
            'text_only': 'Text Only',
            'label_only': 'Label Only',
            'text_label': 'Text+Label'
        }

        for idx, (model_name, results) in enumerate(model_results.items()):
            values = [results.get(k, 0) for k in metric_keys]
            values += values[:1]

            display_name = model_display_names.get(model_name, model_name)

            ax.plot(angles, values, 'o-', linewidth=2,
                   label=display_name, color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'radar_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Radar chart saved: {save_path}")

        return fig


class TrainingCurveVisualizer:
    """
    Training Curve Visualizer - Show curriculum learning three stages
    """

    def __init__(self, save_dir: str = './outputs/figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_curriculum_learning_curves(self,
                                         stage1_history: Dict[str, List[float]],
                                         stage2_history: Dict[str, List[float]],
                                         stage3_history: Dict[str, List[float]],
                                         save_path: str = None) -> plt.Figure:
        """
        Plot curriculum learning three-stage training curves
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        stages = [
            ('Stage 1: Single-Modal Pre-training', stage1_history, '#3498db'),
            ('Stage 2: Dual-Modal Interaction', stage2_history, '#2ecc71'),
            ('Stage 3: Tri-Modal Fusion', stage3_history, '#e74c3c')
        ]

        for ax, (title, history, color) in zip(axes, stages):
            if 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                ax.plot(epochs, history['train_loss'], 'o-', color=color,
                       label='Train Loss', linewidth=2, markersize=4)
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                ax.plot(epochs, history['val_loss'], 's--', color=color,
                       alpha=0.7, label='Val Loss', linewidth=2, markersize=4)
            if 'val_auc' in history:
                ax2 = ax.twinx()
                epochs = range(1, len(history['val_auc']) + 1)
                ax2.plot(epochs, history['val_auc'], '^-', color='purple',
                        label='Val AUC', linewidth=2, markersize=4)
                ax2.set_ylabel('AUC', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')
                ax2.set_ylim(0.5, 1.0)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Curriculum Learning Three-Stage Training Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'curriculum_learning_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Curriculum learning curves saved: {save_path}")

        return fig

    def plot_feature_importance(self,
                                 feature_names: List[str],
                                 importance_scores: np.ndarray,
                                 top_k: int = 15,
                                 save_path: str = None) -> plt.Figure:
        """
        Plot structured feature importance bar chart
        """
        indices = np.argsort(importance_scores)[::-1][:top_k]
        top_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
        top_scores = importance_scores[indices]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_scores)))
        bars = ax.barh(range(len(top_names)), top_scores, color=colors)

        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Structured Feature Importance Top-{top_k}', fontsize=14, fontweight='bold')

        for bar, score in zip(bars, top_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', va='center', fontsize=9)

        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Feature importance figure saved: {save_path}")

        return fig


# ============================================================
# Statistical Analysis
# ============================================================

from scipy import stats

class StatisticalAnalyzer:
    """Statistical Significance Analyzer"""

    def __init__(self, save_dir: str = './outputs/figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def paired_t_test(self, scores1: List[float], scores2: List[float],
                      name1: str = "Model1", name2: str = "Model2"):
        """Paired t-test"""
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        result = {
            'model1': name1,
            'model2': name2,
            'mean1': np.mean(scores1),
            'mean2': np.mean(scores2),
            'std1': np.std(scores1),
            'std2': np.std(scores2),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        print(f"\n📊 Paired t-test: {name1} vs {name2}")
        print(f"  {name1}: {result['mean1']:.4f} ± {result['std1']:.4f}")
        print(f"  {name2}: {result['mean2']:.4f} ± {result['std2']:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if result['significant'] else 'No'}")

        return result

    def wilcoxon_test(self, scores1: List[float], scores2: List[float],
                      name1: str = "Model1", name2: str = "Model2"):
        """Wilcoxon signed-rank test (non-parametric)"""
        stat, p_value = stats.wilcoxon(scores1, scores2)

        result = {
            'model1': name1,
            'model2': name2,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        print(f"\n📊 Wilcoxon test: {name1} vs {name2}")
        print(f"  statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if result['significant'] else 'No'}")

        return result

    def generate_significance_table(self,
                                    all_results: Dict[str, List[float]],
                                    baseline_name: str = "Ours (Full)",
                                    save_path: str = None):
        """
        Generate significance test results table

        Args:
            all_results: Results from multiple runs for all models
                Format: {'model_name': [score1, score2, ...], ...}
            baseline_name: Name of baseline model for comparison
            save_path: Path to save results

        Returns:
            DataFrame with significance test results
        """
        if baseline_name not in all_results:
            print(f"Warning: Baseline model {baseline_name} not found")
            return None

        baseline_scores = all_results[baseline_name]
        results = []

        for model_name, scores in all_results.items():
            if model_name == baseline_name:
                continue

            if len(scores) != len(baseline_scores):
                print(f"Warning: {model_name} sample count mismatch, skipping")
                continue

            t_stat, p_value = stats.ttest_rel(baseline_scores, scores)

            results.append({
                'Model': model_name,
                'Mean': np.mean(scores),
                'Std': np.std(scores),
                'p-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('Mean', ascending=False)

        # Print table
        print("\n" + "=" * 60)
        print(f"Significance Test Results (Baseline: {baseline_name})")
        print("=" * 60)
        print(df.to_string(index=False))

        # Save to CSV
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'significance_test.csv')
        df.to_csv(save_path, index=False)
        print(f"\n✅ Results saved: {save_path}")

        # Generate LaTeX table
        latex_path = save_path.replace('.csv', '.tex')
        latex_content = df.to_latex(index=False, caption='Statistical Significance Test Results',
                                    label='tab:significance')
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        print(f"✅ LaTeX table saved: {latex_path}")

        return df
    def plot_confidence_intervals(self,
                                   results: Dict[str, Dict[str, float]],
                                   metric: str = 'auc',
                                   save_path: str = None):
        """Plot confidence intervals"""
        fig, ax = plt.subplots(figsize=(12, 6))

        models = list(results.keys())
        means = [results[m].get(metric, results[m].get('mean', 0)) for m in models]
        stds = [results[m].get(f'{metric}_std', results[m].get('std', 0.01)) for m in models]

        ci_95 = [1.96 * s for s in stds]

        colors = ['#e74c3c' if 'Ours' in m or 'full' in m.lower() else '#3498db' for m in models]

        y_pos = np.arange(len(models))
        ax.barh(y_pos, means, xerr=ci_95, color=colors, alpha=0.7, capsize=5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel(f'{metric.upper()} (95% CI)', fontsize=12)
        ax.set_title('Model Performance Comparison with Confidence Intervals', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for i, (mean, ci) in enumerate(zip(means, ci_95)):
            ax.text(mean + ci + 0.01, i, f'{mean:.4f}±{ci:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confidence_intervals.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confidence intervals figure saved: {save_path}")

        return fig


if __name__ == "__main__":
    print("Testing visualization tools...")

    # Create sample data
    attention_weights = {
        'text_to_label': torch.randn(1, 8, 10, 10),
        'label_to_text': torch.randn(1, 8, 10, 10)
    }

    # Test attention visualization
    visualizer = AttentionVisualizer()
    visualizer.visualize_cross_modal_attention(
        attention_weights,
        sample_text="Test complaint text content"
    )

    # Test feature importance visualization
    feature_viz = FeatureImportanceVisualizer()
    feature_names = ['Tenure', 'Monthly_Spend', 'Complaint_Count', 'Satisfaction', 'Plan_Type']
    feature_importance = np.random.rand(5)
    feature_viz.visualize_structured_features(feature_names, feature_importance)

    print("Visualization test completed!")