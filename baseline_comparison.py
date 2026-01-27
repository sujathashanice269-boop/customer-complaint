"""
Baseline Comparison Experiments - baseline_comparison.py
Compare with traditional ML and other deep learning methods

Usage: python baseline_comparison.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import ComplaintDataProcessor


class BaselineExperiments:
    """Baseline Experiments Class"""

    def __init__(self, data_file: str = '小案例ai问询.xlsx'):
        self.config = Config()

        # 设置全局随机种子
        self.global_seed = 42
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.global_seed)
        print(f"✓ 全局随机种子设置为: {self.global_seed}")
        self.data_file = data_file
        self.results = {}

        # Load data
        print("📂 Loading data...")
        if data_file.endswith('.xlsx'):
            self.df = pd.read_excel(data_file)
        else:
            self.df = pd.read_csv(data_file)

        print(f"Dataset size: {len(self.df)}")

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data"""
        # Text
        self.texts = self.df['biz_cntt'].fillna('').astype(str).tolist()

        # Labels
        self.labels = self.df['Complaint label'].fillna('').astype(str).tolist()

        # Target variable
        self.targets = self.df['Repeat complaint'].values

        # Structured features - 修复版本
        col_names = self.df.columns.tolist()
        exclude_cols = {'new_code', 'biz_cntt', 'Complaint label', 'Repeat complaint'}

        if 'Complaint label' in col_names and 'Repeat complaint' in col_names:
            label_idx = col_names.index('Complaint label')
            target_idx = col_names.index('Repeat complaint')
            struct_cols = col_names[label_idx + 1: target_idx]
        else:
            struct_cols = col_names[3:56]

        # 排除非特征列
        struct_cols = [col for col in struct_cols if col not in exclude_cols][:53]

        print(f"结构化特征列数: {len(struct_cols)}")
        self.struct_features = self.df[struct_cols].fillna(0).values

        # Split dataset
        indices = np.arange(len(self.df))
        self.train_idx, self.test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=self.targets
        )

        print(f"Train set: {len(self.train_idx)}, Test set: {len(self.test_idx)}")

    def _evaluate(self, y_true, y_pred, y_prob=None):
        """Evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        return metrics

    def run_tfidf_lr(self):
        """TF-IDF + Logistic Regression"""
        print("\n📊 Experiment: TF-IDF + Logistic Regression")
        # 设置独立种子
        np.random.seed(101)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(self.texts)

        X_train = X[self.train_idx]
        X_test = X[self.test_idx]
        y_train = self.targets[self.train_idx]
        y_test = self.targets[self.test_idx]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._evaluate(y_test, y_pred, y_prob)
        self.results['TF-IDF + LR'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_tfidf_rf(self):
        """TF-IDF + Random Forest"""
        print("\n📊 Experiment: TF-IDF + Random Forest")
        # 设置独立种子
        np.random.seed(102)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(self.texts)

        X_train = X[self.train_idx]
        X_test = X[self.test_idx]
        y_train = self.targets[self.train_idx]
        y_test = self.targets[self.test_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._evaluate(y_test, y_pred, y_prob)
        self.results['TF-IDF + RF'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_tfidf_gbdt(self):
        """TF-IDF + GBDT"""
        print("\n📊 Experiment: TF-IDF + GBDT")
        # 设置独立种子
        np.random.seed(103)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(self.texts)

        X_train = X[self.train_idx].toarray()
        X_test = X[self.test_idx].toarray()
        y_train = self.targets[self.train_idx]
        y_test = self.targets[self.test_idx]

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._evaluate(y_test, y_pred, y_prob)
        self.results['TF-IDF + GBDT'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_struct_only_lr(self):
        """Structured Features Only + Logistic Regression"""
        print("\n📊 Experiment: Structured Features + LR")
        # 设置独立种子
        np.random.seed(104)
        X_train = self.struct_features[self.train_idx]
        X_test = self.struct_features[self.test_idx]
        y_train = self.targets[self.train_idx]
        y_test = self.targets[self.test_idx]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._evaluate(y_test, y_pred, y_prob)
        self.results['Struct + LR'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_struct_only_rf(self):
        """Structured Features Only + Random Forest"""
        print("\n📊 Experiment: Structured Features + RF")
        # 设置独立种子
        np.random.seed(105)
        X_train = self.struct_features[self.train_idx]
        X_test = self.struct_features[self.test_idx]
        y_train = self.targets[self.train_idx]
        y_test = self.targets[self.test_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._evaluate(y_test, y_pred, y_prob)
        self.results['Struct + RF'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_tfidf_struct_lr(self):
        """TF-IDF + Structured Features + Logistic Regression"""
        print("\n📊 Experiment: TF-IDF + Struct + LR")
        # 设置独立种子
        np.random.seed(106)
        vectorizer = TfidfVectorizer(max_features=3000)
        X_text = vectorizer.fit_transform(self.texts).toarray()

        # Combine features
        X = np.hstack([X_text, self.struct_features])

        X_train = X[self.train_idx]
        X_test = X[self.test_idx]
        y_train = self.targets[self.train_idx]
        y_test = self.targets[self.test_idx]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._evaluate(y_test, y_pred, y_prob)
        self.results['TF-IDF + Struct + LR'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_mlp(self):
        """TF-IDF + Structured + MLP"""
        print("\n📊 Experiment: TF-IDF + Struct + MLP")
        # 设置独立种子
        np.random.seed(107)
        vectorizer = TfidfVectorizer(max_features=3000)
        X_text = vectorizer.fit_transform(self.texts).toarray()
        X = np.hstack([X_text, self.struct_features])

        X_train = X[self.train_idx]
        X_test = X[self.test_idx]
        y_train = self.targets[self.train_idx]
        y_test = self.targets[self.test_idx]

        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._evaluate(y_test, y_pred, y_prob)
        self.results['TF-IDF + Struct + MLP'] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def run_all(self):
        """Run all baseline experiments"""
        print("\n" + "="*60)
        print("🔬 Baseline Comparison Experiments")
        print("="*60)

        self.run_tfidf_lr()
        self.run_tfidf_rf()
        self.run_tfidf_gbdt()
        self.run_struct_only_lr()
        self.run_struct_only_rf()
        self.run_tfidf_struct_lr()
        self.run_mlp()

        # Load our model results (if exists)
        if os.path.exists('ablation_results.json'):
            with open('ablation_results.json', 'r') as f:
                ablation = json.load(f)
            if 'full_model' in ablation:
                self.results['Ours (Full)'] = ablation['full_model']

        # Print summary
        self._print_summary()

        # Save results
        self._save_results()

        return self.results

    def _print_summary(self):
        """Print summary results"""
        print("\n" + "="*60)
        print("📊 Baseline Comparison Results Summary")
        print("="*60)
        print(f"\n{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        print("-" * 85)

        # Sort by AUC
        sorted_results = sorted(self.results.items(),
                               key=lambda x: x[1].get('auc', 0), reverse=True)

        for name, metrics in sorted_results:
            print(f"{name:<25} {metrics.get('accuracy', 0):<12.4f} "
                  f"{metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} "
                  f"{metrics.get('f1', 0):<12.4f} {metrics.get('auc', 0):<12.4f}")

        print("-" * 85)

        # Calculate improvement
        if 'Ours (Full)' in self.results:
            our_auc = self.results['Ours (Full)'].get('auc', 0)
            best_baseline_auc = max([v.get('auc', 0) for k, v in self.results.items()
                                    if k != 'Ours (Full)'])
            improvement = (our_auc - best_baseline_auc) / best_baseline_auc * 100
            print(f"\n🎯 Our method improvement over best baseline: {improvement:.2f}%")

    def _save_results(self):
        """Save results"""
        with open('baseline_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved to: baseline_results.json")

        # Generate LaTeX table
        self._generate_latex_table()

    def _generate_latex_table(self):
        """Generate LaTeX table"""
        latex = """
\\begin{table}[htbp]
\\centering
\\caption{Baseline Methods Comparison Results}
\\label{tab:baseline}
\\begin{tabular}{lccccc}
\\toprule
Method & Accuracy & Precision & Recall & F1 & AUC \\\\
\\midrule
"""
        sorted_results = sorted(self.results.items(),
                               key=lambda x: x[1].get('auc', 0), reverse=True)

        for name, metrics in sorted_results:
            latex += f"{name} & {metrics.get('accuracy', 0):.4f} & "
            latex += f"{metrics.get('precision', 0):.4f} & {metrics.get('recall', 0):.4f} & "
            latex += f"{metrics.get('f1', 0):.4f} & {metrics.get('auc', 0):.4f} \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        with open('baseline_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"✅ LaTeX table saved to: baseline_table.tex")


def main():
    experiments = BaselineExperiments(data_file='小案例ai问询.xlsx')
    experiments.run_all()


if __name__ == "__main__":
    main()