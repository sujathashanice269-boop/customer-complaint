"""
Interpretability Analysis Script - interpretability_analysis.py
For generating visualization figures and analysis reports for papers

Features:
1. Cross-modal attention heatmaps
2. Typical case decision tracing
3. Modality contribution analysis
4. Feature importance ranking

Usage:
    python interpretability_analysis.py --model_path ./models/best_model.pth --mode all
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from model import MultiModalComplaintModel
from data_processor import ComplaintDataProcessor
from config import Config
from visualization import (
    EnhancedAttentionVisualizer,
    ModalityContributionAnalyzer,
    TrainingCurveVisualizer
)


def load_struct_feature_names_from_excel(data_file: str, start_col: str = 'D', end_col: str = 'BD') -> List[str]:
    """
    从Excel数据集中读取结构化特征名称（D列~BD列的首行）

    Args:
        data_file: Excel文件路径
        start_col: 起始列 (默认'D')
        end_col: 结束列 (默认'BD')

    Returns:
        特征名称列表
    """
    try:
        import pandas as pd

        # 读取Excel文件的第一行
        df = pd.read_excel(data_file, nrows=1)

        # 获取列索引
        def col_to_idx(col):
            result = 0
            for char in col.upper():
                result = result * 26 + (ord(char) - ord('A') + 1)
            return result - 1

        start_idx = col_to_idx(start_col)
        end_idx = col_to_idx(end_col)

        # 获取列名
        all_cols = list(df.columns)
        if end_idx >= len(all_cols):
            end_idx = len(all_cols) - 1

        feature_names = all_cols[start_idx:end_idx + 1]

        # 清理列名（去除空格等）
        feature_names = [str(name).strip() for name in feature_names]

        print(f"✅ 从数据集加载了 {len(feature_names)} 个结构化特征名称")
        return feature_names

    except Exception as e:
        print(f"⚠️ 无法从数据集加载特征名称: {e}")
        print("   将使用默认特征名称")
        return None


# 默认的结构化特征名称（当无法从数据集加载时使用）
# 修正后的结构化特征名称（与您的数据完全匹配）
DEFAULT_STRUCT_FEATURE_NAMES = [
    'Complaint channel','Credit star','Global tier','Upgd-cmplt','Svy-timegap','Svy_timing','Ur-timegap', 'Ur-timing',
    'Ur-recommend accept','Svy-transparency','Svy-olduser echannel','Svy-policy','Svy-newuser echannel','Svy-newuser store',
    'Svy-promotion','Svy-mobile network','Svy-performance','Svy-service usage','Svy-newuser hotline','Svy-expectation',
    'Svy-olduser hotline','Svy-olduser store','Svy-net complaint','Svy-nps','Svy-channel complaint','Svy-other',
    'Svy_no complaint','Svy-marketing_complaint','Svy-professionalism','Svy-timeliness','Svy-complaint result','Svy-complaint sat',
    'Phone status','Package brand','Age','Online month','Vip','No-disturb','Dual sim-susp','Phone Brand','Campus user','Volte potential',
    'Price sensitive','No-Broadband','Competitor broadband','Tencent king-applied','Tencent king-potential','Migrant worker',
    'Other rejoin','Back rejoin','Interviewee','Customer segment','Gender'
]

class InterpretabilityAnalyzer:
    """
    Interpretability Analyzer - Core Class
    For extracting model attention weights, conducting case analysis, etc.
    """

    def __init__(self,
                 model_path: str,
                 pretrained_path: str = None,
                 config: Config = None,
                 device: str = None):
        """
        Initialize analyzer

        Args:
            model_path: Path to trained model
            pretrained_path: Path to pretrained model
            config: Configuration object
            device: Device to use
        """
        self.config = config or Config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load processor
        self.processor = ComplaintDataProcessor(
            config=self.config,
            user_dict_file=self.config.data.user_dict_file
        )

        # Try to load pretrained processor
        if pretrained_path is None:
            pretrained_path = self.config.training.pretrain_save_dir

        processor_path = os.path.join(pretrained_path, 'processor.pkl') if pretrained_path else None
        if processor_path and os.path.exists(processor_path):
            print(f"✅ Loading processor: {processor_path}")
            self.processor.load(processor_path)

        # Get vocabulary size
        vocab_size = len(self.processor.node_to_id) if self.processor.node_to_id else 1000

        # Load model
        print(f"📦 Loading model: {model_path}")
        self.model = MultiModalComplaintModel(
            config=self.config,
            vocab_size=vocab_size,
            mode='full',
            pretrained_path=pretrained_path
        )

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Model weights loaded successfully")
        else:
            print(f"⚠️ Model file not found: {model_path}, using random initialization")

        self.model.to(self.device)
        self.model.eval()

        # Initialize visualizers
        self.attention_viz = EnhancedAttentionVisualizer(save_dir='./outputs/figures')
        self.contribution_analyzer = ModalityContributionAnalyzer(save_dir='./outputs/figures')
        self.curve_viz = TrainingCurveVisualizer(save_dir='./outputs/figures')
        # 【新增】从数据集加载结构化特征名称（D列~BD列的首行）
        data_file = self.config.training.data_file
        loaded_names = load_struct_feature_names_from_excel(data_file, 'D', 'BD')
        if loaded_names:
            self.struct_feature_names = loaded_names
        else:
            # 使用默认特征名称
            self.struct_feature_names = DEFAULT_STRUCT_FEATURE_NAMES.copy()
        print(f"✅ Analyzer initialized, using device: {self.device}")

    def extract_attention_weights(self,
                                    text: str,
                                    label: str,
                                    struct_features: List[float]) -> Tuple[Dict, Dict]:
        """
        Extract attention weights for a single sample
        """
        # Text encoding
        text_encoding = self.processor.tokenizer(
            text,
            max_length=self.config.model.bert_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = text_encoding['input_ids'].to(self.device)
        attention_mask = text_encoding['attention_mask'].to(self.device)

        # Label encoding
        node_ids, edges, levels = self.processor.encode_label_path_as_graph(label)

        # Structured features
        struct_tensor = torch.tensor(struct_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits, attention_weights = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_ids_list=[node_ids],
                edges_list=[edges],
                node_levels_list=[levels],
                struct_features=struct_tensor,
                return_attention=True
            )

            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0][pred_class].item()

        prediction_info = {
            'prediction': pred_class,
            'confidence': confidence,
            'prob_non_repeat': probs[0][0].item(),
            'prob_repeat': probs[0][1].item()
        }

        return attention_weights, prediction_info

    def analyze_single_case(self,
                            text: str,
                            label: str,
                            struct_features: List[float],
                            true_label: int = None,
                            case_id: str = "case_001",
                            new_code: str = None) -> Dict:
        """Analyze a single case and generate visualization"""
        print(f"\n{'='*60}")
        print(f"📊 Analyzing case: {case_id}")
        if new_code:
            print(f"📌 new_code: {new_code}")
        print("="*60)

        attention_weights, pred_info = self.extract_attention_weights(
            text, label, struct_features
        )

        self.attention_viz.plot_cross_modal_attention_heatmap(
            attention_weights=attention_weights,
            sample_id=case_id,
            label_path=label,
            tokenizer=self.processor.tokenizer,
            text=text,
            new_code=new_code,
            struct_feature_names=self.struct_feature_names,
            save_path=f'./outputs/figures/attention_{case_id}.png'
        )

        self.attention_viz.plot_attention_with_text(
            attention_weights=attention_weights,
            text=text,
            label_path=label,
            prediction=pred_info['prediction'],
            confidence=pred_info['confidence'],
            true_label=true_label,
            sample_id=case_id,
            new_code=new_code,
            tokenizer=self.processor.tokenizer,
            struct_feature_names=self.struct_feature_names,
            save_path=f'./outputs/figures/case_study_{case_id}.png'
        )

        result = {
            'case_id': case_id,
            'new_code': new_code,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'label': label,
            'prediction': 'Repeat Complaint' if pred_info['prediction'] == 1 else 'Non-Repeat',
            'confidence': pred_info['confidence'],
            'true_label': true_label,
            'is_correct': pred_info['prediction'] == true_label if true_label is not None else None
        }

        print(f"  Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
        if true_label is not None:
            print(f"  True Label: {'Repeat Complaint' if true_label == 1 else 'Non-Repeat'}")
            print(f"  Status: {'✅ Correct' if result['is_correct'] else '❌ Incorrect'}")

        return result

    def find_contrastive_cases(self,
                               data_file: str,
                               n_cases: int = 5) -> List[Dict]:
        """Find contrastive cases - 修复版本"""
        print(f"\n🔍 Searching for contrastive cases...")

        if data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        else:
            df = pd.read_csv(data_file)

        contrastive_cases = []
        col_names = df.columns.tolist()

        exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Repeat complaint'}

        if 'Complaint label' in col_names and 'Repeat complaint' in col_names:
            label_idx = col_names.index('Complaint label')
            target_idx = col_names.index('Repeat complaint')
            struct_cols = col_names[label_idx + 1: target_idx]
        else:
            struct_cols = col_names[3:56]

        struct_cols = [col for col in struct_cols if col not in exclude_cols][:53]
        print(f"  结构化特征列数: {len(struct_cols)}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning samples"):
            if len(contrastive_cases) >= n_cases:
                break

            new_code = str(row.get('new_code', f'row_{idx}'))
            text = str(row.get('biz_cntt', ''))
            label = str(row.get('Complaint label', ''))
            true_label = row.get('Repeat complaint', 0)

            struct_features = []
            for col in struct_cols:
                try:
                    val = pd.to_numeric(row.get(col, 0), errors='coerce')
                    struct_features.append(0 if pd.isna(val) else val)
                except:
                    struct_features.append(0)

            while len(struct_features) < 53:
                struct_features.append(0)

            try:
                _, pred_info = self.extract_attention_weights(text, label, struct_features)
                full_pred = pred_info['prediction']
                full_correct = (full_pred == true_label)

                if full_correct:
                    print(f"\n📌 Found case: new_code = {new_code}")
                    print(
                        f"   True: {'Repeat' if true_label == 1 else 'Non-Repeat'}, Conf: {pred_info['confidence']:.2%}")

                    contrastive_cases.append({
                        'idx': idx,
                        'new_code': new_code,
                        'text': text,
                        'label': label,
                        'struct_features': struct_features,
                        'true_label': true_label,
                        'full_pred': full_pred,
                        'confidence': pred_info['confidence']
                    })
            except Exception as e:
                continue

        print(f"\n{'=' * 60}")
        print("📋 Selected Cases Summary (new_code):")
        print("=" * 60)
        for i, case in enumerate(contrastive_cases[:n_cases]):
            print(f"  Case {i + 1}: new_code = {case['new_code']}")
        print("=" * 60 + "\n")

        return contrastive_cases[:n_cases]

    def extract_feature_importance(self) -> Tuple[List[str], np.ndarray]:
        """
        Extract structured feature importance scores
        """
        print("\n📊 Extracting feature importance...")

        if hasattr(self.model, 'feature_importance') and self.model.feature_importance is not None:
            importance = self.model.feature_importance.detach().cpu().numpy()
            importance = np.exp(importance) / np.sum(np.exp(importance))
        else:
            importance = np.ones(53) / 53

        # Feature names (English)
        feature_names = [
            'Tenure', 'Monthly_Spend', 'Plan_Type', 'Complaint_Count', 'Satisfaction_Score',
            'Recent_Spend_Change', 'Account_Balance', 'Data_Usage', 'Call_Duration', 'SMS_Count',
            'VAS_Count', 'Bill_Anomaly', 'Overdue_Count', 'Payment_Method', 'Channel_Preference',
            'Contract_Status', 'Device_Type', 'Brand_Tier', 'Activity_Level', 'Social_Activity',
            'Complaint_Interval', 'Resolution_Time', 'CS_Contact', 'Escalation_Count', 'Compensation_Record',
            'Credit_Score', 'Age_Group', 'Gender', 'Region', 'Network_Quality',
            'Signal_Strength', 'Drop_Rate', 'Speed', 'Coverage', 'Fault_Record',
            'Repair_Count', 'Maintenance_Response', 'Device_Status', 'Usage_Pattern', 'Peak_Hours',
            'Data_Overage', 'Plan_Match', 'Price_Sensitivity', 'Promo_Response', 'Loyalty',
            'Churn_Risk', 'Value_Tier', 'Service_Rating', 'Issue_Type', 'Urgency',
            'Repeat_Flag', 'Multi_Channel', 'Feature_53'
        ]

        if len(feature_names) < len(importance):
            feature_names.extend([f'Feature_{i}' for i in range(len(feature_names), len(importance))])

        return feature_names[:len(importance)], importance

    def run_full_analysis(self,
                          data_file: str = None,
                          ablation_results_file: str = None,
                          n_cases: int = 3):
        """
        Run complete interpretability analysis
        """
        print("\n" + "="*60)
        print("🔬 Starting Complete Interpretability Analysis")
        print("="*60)

        os.makedirs('./outputs/figures', exist_ok=True)
        os.makedirs('./outputs/reports', exist_ok=True)

        results = {}

        # 1. Feature importance analysis
        print("\n📊 [1/4] Feature Importance Analysis")
        feature_names, importance = self.extract_feature_importance()
        self.curve_viz.plot_feature_importance(
            feature_names=feature_names,
            importance_scores=importance,
            top_k=15,
            save_path='./outputs/figures/feature_importance.png'
        )
        results['feature_importance'] = {
            'names': feature_names[:15],
            'scores': importance[:15].tolist()
        }

        # 2. Ablation comparison figure
        print("\n📊 [2/4] Ablation Study Comparison")
        if ablation_results_file and os.path.exists(ablation_results_file):
            with open(ablation_results_file, 'r') as f:
                ablation_results = json.load(f)

            self.contribution_analyzer.plot_ablation_comparison(
                ablation_results=ablation_results,
                save_path='./outputs/figures/ablation_comparison.png'
            )

            # Calculate modality contributions
            full_auc = ablation_results.get('full_model', {}).get('auc', 0.9)
            text_auc = ablation_results.get('text_only', {}).get('auc', 0.7)
            label_auc = ablation_results.get('label_only', {}).get('auc', 0.6)
            struct_auc = ablation_results.get('struct_only', {}).get('auc', 0.55)

            total = text_auc + label_auc + struct_auc
            contributions = {
                'Text': text_auc / total,
                'Label': label_auc / total,
                'Structured': struct_auc / total
            }

            self.contribution_analyzer.plot_modality_contribution_pie(
                contributions=contributions,
                save_path='./outputs/figures/modality_contribution.png'
            )

            # Radar chart
            selected_models = {k: v for k, v in ablation_results.items()
                             if k in ['full_model', 'text_only', 'label_only', 'text_label']}
            if selected_models:
                self.contribution_analyzer.plot_radar_comparison(
                    model_results=selected_models,
                    save_path='./outputs/figures/radar_comparison.png'
                )

            results['ablation'] = ablation_results
        else:
            print("  ⚠️ Ablation results file not found, skipping")

        # 3. Typical case analysis
        print("\n📊 [3/4] Typical Case Analysis")
        if data_file and os.path.exists(data_file):
            cases = self.find_contrastive_cases(data_file, n_cases=n_cases)

            case_results = []
            for i, case in enumerate(cases):
                new_code = case.get('new_code', f'unknown_{i}')

                result = self.analyze_single_case(
                    text=case['text'],
                    label=case['label'],
                    struct_features=case['struct_features'],
                    true_label=case['true_label'],
                    case_id=f"case_{i + 1:03d}",
                    new_code=new_code
                )
                case_results.append(result)

            results['case_studies'] = case_results
        else:
            print("  ⚠️ Data file not found, skipping case analysis")

        # 4. Save analysis report
        print("\n📊 [4/4] Generating Analysis Report")
        report_path = './outputs/reports/interpretability_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"  ✅ Report saved: {report_path}")

        print("\n" + "="*60)
        print("✅ Interpretability Analysis Complete!")
        print("="*60)
        print("\n📁 Output files:")
        print("  figures/")
        for f in os.listdir('./outputs/figures'):
            if f.endswith('.png'):
                print(f"    - {f}")
        print("  reports/")
        print(f"    - interpretability_report.json")

        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interpretability Analysis Tool')
    parser.add_argument('--model_path', type=str,
                        default='./models/best_full_model.pth',
                        help='Model path')
    parser.add_argument('--pretrained_path', type=str,
                        default='./pretrained_complaint_bert_improved',
                        help='Pretrained model path')
    parser.add_argument('--data_file', type=str,
                        default='小案例ai问询.xlsx',
                        help='Data file path')
    parser.add_argument('--ablation_results', type=str,
                        default='ablation_results.json',
                        help='Ablation results file')
    parser.add_argument('--mode', type=str,
                        choices=['all', 'attention', 'cases', 'features'],
                        default='all',
                        help='Analysis mode')
    parser.add_argument('--n_cases', type=int, default=3,
                        help='Number of cases to analyze')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔬 Interpretability Analysis Tool")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Data file: {args.data_file}")
    print(f"Analysis mode: {args.mode}")
    print("="*60)

    try:
        analyzer = InterpretabilityAnalyzer(
            model_path=args.model_path,
            pretrained_path=args.pretrained_path
        )

        if args.mode == 'all':
            analyzer.run_full_analysis(
                data_file=args.data_file,
                ablation_results_file=args.ablation_results,
                n_cases=args.n_cases
            )
        elif args.mode == 'features':
            feature_names, importance = analyzer.extract_feature_importance()
            analyzer.curve_viz.plot_feature_importance(
                feature_names=feature_names,
                importance_scores=importance
            )
        elif args.mode == 'cases':
            cases = analyzer.find_contrastive_cases(args.data_file, n_cases=args.n_cases)
            for i, case in enumerate(cases):
                analyzer.analyze_single_case(
                    text=case['text'],
                    label=case['label'],
                    struct_features=case['struct_features'],
                    true_label=case['true_label'],
                    case_id=f"case_{i+1:03d}"
                )

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())