"""
FedCF-IoT: Visualization Module (FIXED)
=======================================
Generates all figures for the experiments in publication-quality format.

FIXES:
- Fixed "unhashable type: 'dict'" error in plot_baseline_comparison
- Added safety checks for dict iteration
- Added try-except wrappers for robust figure generation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import os

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#17A2B8',
    'dark': '#343A40',
    'methods': ['#2E86AB', '#A23B72', '#28A745', '#F18F01', '#C73E1D',
                '#17A2B8', '#6C757D', '#9B59B6', '#1ABC9C', '#E74C3C']
}


class ExperimentVisualizer:
    """Generates all visualization for FedCF-IoT experiments"""

    def __init__(self, results: Dict, output_dir: str, dataset_name: str = ""):
        self.results = results
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

    def _safe_plot(self, plot_func, fig_name):
        """Safely execute plot function with error handling"""
        try:
            plot_func()
            print(f"  [✓] {fig_name}")
            return fig_name
        except Exception as e:
            print(f"  [✗] {fig_name} - Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_all(self):
        """Generate all figures with error handling"""
        print("\\n" + "="*60)
        print("GENERATING FIGURES")
        print("="*60)

        generated = []

        # 1. Class Distribution
        if 'dataset_info' in self.results:
            fig = self._safe_plot(self.plot_class_distribution, "class_distribution.png")
            if fig: generated.append(fig)

        # 2. Training Curve
        if 'exp1_training' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_training_curve, "training_curve.png")
            if fig: generated.append(fig)

            # 3. Confusion Matrix
            if 'classification_metrics' in self.results['experiments']['exp1_training']:
                fig = self._safe_plot(self.plot_confusion_matrix, "confusion_matrix.png")
                if fig: generated.append(fig)

                fig = self._safe_plot(self.plot_roc_curves, "roc_curves.png")
                if fig: generated.append(fig)

            # 4. Baseline Comparison
            if 'baseline_comparison' in self.results['experiments']['exp1_training']:
                fig = self._safe_plot(self.plot_baseline_comparison, "baseline_comparison.png")
                if fig: generated.append(fig)

        # 5. CF Methods Comparison
        if 'exp2_cf_comparison' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_cf_comparison, "cf_methods_comparison.png")
            if fig: generated.append(fig)

            fig = self._safe_plot(self.plot_cf_comparison_detailed, "cf_methods_detailed.png")
            if fig: generated.append(fig)

        # 6. Privacy-Utility Tradeoff
        if 'exp3_privacy' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_privacy_utility, "privacy_utility_tradeoff.png")
            if fig: generated.append(fig)

        # 7. FL Aggregation
        if 'exp4_fl_aggregation' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_fl_aggregation, "fl_aggregation.png")
            if fig: generated.append(fig)

        # 8. Scalability
        if 'exp5_scalability' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_scalability, "scalability.png")
            if fig: generated.append(fig)

        # 9. Non-IID Heterogeneity
        if 'exp6_heterogeneity' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_heterogeneity, "heterogeneity.png")
            if fig: generated.append(fig)

        # 10. Multi-class Analysis
        if 'exp7_multiclass' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_multiclass_analysis, "multiclass_analysis.png")
            if fig: generated.append(fig)

        # 11. Ablation Study
        if 'exp8_ablation' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_ablation, "ablation_study.png")
            if fig: generated.append(fig)

        # 12. Latency Analysis
        if 'exp9_latency' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_latency, "latency_analysis.png")
            if fig: generated.append(fig)

        # 13. Privacy Attack (MIA)
        if 'exp10_privacy_attack' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_mia_results, "mia_results.png")
            if fig: generated.append(fig)

        # 14. Case Study Feature Importance
        if 'exp11_case_study' in self.results.get('experiments', {}):
            fig = self._safe_plot(self.plot_cf_feature_importance, "cf_feature_importance.png")
            if fig: generated.append(fig)

        print(f"\\nGenerated {len(generated)} figures in {self.figures_dir}/")
        return generated

    def plot_class_distribution(self):
        """Plot class distribution pie chart and bar chart"""
        ds_info = self.results.get('dataset_info', {})
        class_dist = ds_info.get('class_distribution', {})

        if not class_dist:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = list(class_dist.keys())
        values = list(class_dist.values())
        total = sum(values)
        percentages = [v/total*100 for v in values]

        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = axes[0].pie(
            values, labels=None, autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
            colors=colors, startangle=90
        )
        axes[0].set_title('Class Distribution', fontweight='bold')
        axes[0].legend(wedges, labels, title="Classes", loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))

        bars = axes[1].bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Sample Count per Class', fontweight='bold')
        axes[1].set_yscale('log')

        for bar, val, pct in zip(bars, values, percentages):
            axes[1].annotate(f'{val:,}\\n({pct:.1f}%)',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'class_distribution.png'))
        plt.close()

    def plot_training_curve(self):
        """Plot training accuracy curve"""
        exp1 = self.results['experiments']['exp1_training']
        history = exp1.get('history', [])

        if not history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history, 'b-o', linewidth=2, markersize=6,
                label='Test Accuracy', color=COLORS['primary'])

        final_acc = history[-1]
        ax.annotate(f'Final: {final_acc:.4f}',
                   xy=(len(history), final_acc),
                   xytext=(len(history)-3, final_acc-0.02),
                   fontsize=12, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray'))

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Model Training Progress - {self.dataset_name}', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(history)*0.95, min(1.0, max(history)*1.02)])

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'training_curve.png'))
        plt.close()

    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        exp1 = self.results['experiments']['exp1_training']
        clf_metrics = exp1.get('classification_metrics', {})
        cm = clf_metrics.get('confusion_matrix', [])

        if not cm:
            return

        cm = np.array(cm)

        per_class = clf_metrics.get('per_class_metrics', {})
        class_names = list(per_class.keys()) if per_class else [f'Class {i}' for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(10, 8))

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Proportion'})

        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(f'Normalized Confusion Matrix - {self.dataset_name}', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_roc_curves(self):
        """Plot per-class performance metrics"""
        exp1 = self.results['experiments']['exp1_training']
        clf_metrics = exp1.get('classification_metrics', {})
        per_class = clf_metrics.get('per_class_metrics', {})

        if not per_class:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        class_names = list(per_class.keys())
        f1_scores = [per_class[c].get('f1', 0) for c in class_names]
        recall_scores = [per_class[c].get('recall', 0) for c in class_names]
        precision_scores = [per_class[c].get('precision', 0) for c in class_names]

        x = np.arange(len(class_names))
        width = 0.25

        ax.bar(x - width, precision_scores, width, label='Precision', color=COLORS['primary'])
        ax.bar(x, recall_scores, width, label='Recall', color=COLORS['secondary'])
        ax.bar(x + width, f1_scores, width, label='F1-Score', color=COLORS['success'])

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Performance Metrics - {self.dataset_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])

        f1_macro = clf_metrics.get('f1_macro', 0)
        ax.axhline(y=f1_macro, color='red', linestyle='--', label=f"Macro F1: {f1_macro:.3f}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'roc_curves.png'))
        plt.close()

    def plot_baseline_comparison(self):
        """Plot baseline model comparison - FIXED"""
        exp1 = self.results['experiments']['exp1_training']
        baselines = exp1.get('baseline_comparison', {})
        clf_metrics = exp1.get('classification_metrics', {})

        if not baselines:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # FIXED: Filter out non-model entries (like comparison_metadata)
        models = ['MLP (Ours)']
        accuracies = [clf_metrics.get('accuracy', 0)]
        f1_macros = [clf_metrics.get('f1_macro', 0)]

        for name, res in baselines.items():
            # Skip non-model entries
            if not isinstance(res, dict):
                continue
            if 'accuracy' not in res and 'f1_macro' not in res:
                continue
            if name in ['comparison_metadata', 'fair_comparison_settings']:
                continue

            models.append(name)
            accuracies.append(res.get('accuracy', 0))
            f1_macros.append(res.get('f1_macro', 0))

        if len(models) <= 1:
            plt.close()
            return

        x = np.arange(len(models))
        colors = [COLORS['primary']] + [COLORS['methods'][i % len(COLORS['methods'])]
                                         for i in range(len(models)-1)]

        # Accuracy comparison
        bars1 = axes[0].bar(x, accuracies, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylim([0, 1.05])

        for bar, val in zip(bars1, accuracies):
            axes[0].annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10)

        # F1-macro comparison
        bars2 = axes[1].bar(x, f1_macros, color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('F1-Score (macro)')
        axes[1].set_title('F1-Score Comparison', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylim([0, 1.05])

        for bar, val in zip(bars2, f1_macros):
            axes[1].annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10)

        plt.suptitle(f'Model Comparison - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'baseline_comparison.png'))
        plt.close()

    def plot_cf_comparison(self):
        """Plot CF methods comparison bar chart - FIXED"""
        exp2 = self.results['experiments']['exp2_cf_comparison']
        methods = exp2.get('methods', {})

        if not methods:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        names = []
        validities = []

        for name, res in methods.items():
            # FIXED: Skip non-method entries
            if not isinstance(res, dict):
                continue
            if 'validity' not in res:
                continue
            names.append(name)
            validities.append(res.get('validity', 0) * 100)

        if not names:
            plt.close()
            return

        colors = [COLORS['methods'][i % len(COLORS['methods'])] for i in range(len(names))]

        bars = ax.bar(names, validities, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Validity (%)')
        ax.set_title(f'Counterfactual Methods Comparison - {self.dataset_name}', fontweight='bold')
        ax.set_ylim([0, 110])

        for bar, val in zip(bars, validities):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cf_methods_comparison.png'))
        plt.close()

    def plot_cf_comparison_detailed(self):
        """Plot detailed CF metrics comparison"""
        exp2 = self.results['experiments']['exp2_cf_comparison']
        methods = exp2.get('methods', {})

        if not methods:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        names = []
        validities = []
        proximities = []
        sparsities = []
        diversities = []

        for name, res in methods.items():
            if not isinstance(res, dict) or 'validity' not in res:
                continue
            names.append(name)
            validities.append(res.get('validity', 0) * 100)
            proximities.append(res.get('proximity', 0))
            sparsities.append(res.get('sparsity', 0))
            diversities.append(res.get('diversity', 0))

        if not names:
            plt.close()
            return

        colors = [COLORS['methods'][i % len(COLORS['methods'])] for i in range(len(names))]

        # Validity
        axes[0, 0].bar(names, validities, color=colors, edgecolor='black')
        axes[0, 0].set_ylabel('Validity (%)')
        axes[0, 0].set_title('Validity (Higher is Better)', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Proximity
        axes[0, 1].bar(names, proximities, color=colors, edgecolor='black')
        axes[0, 1].set_ylabel('Proximity (L1)')
        axes[0, 1].set_title('Proximity (Lower is Better)', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Sparsity
        axes[1, 0].bar(names, sparsities, color=colors, edgecolor='black')
        axes[1, 0].set_ylabel('Sparsity')
        axes[1, 0].set_title('Sparsity (Lower is Better)', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Diversity
        axes[1, 1].bar(names, diversities, color=colors, edgecolor='black')
        axes[1, 1].set_ylabel('Diversity')
        axes[1, 1].set_title('Diversity (Higher is Better)', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle(f'CF Methods Detailed Comparison - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cf_methods_detailed.png'))
        plt.close()

    def plot_privacy_utility(self):
        """Plot privacy-utility tradeoff"""
        exp3 = self.results['experiments']['exp3_privacy']
        epsilon_data = exp3.get('epsilons', {})

        if not epsilon_data:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epsilons = []
        validities = []

        for key, data in sorted(epsilon_data.items(), key=lambda x: float(x[1].get('epsilon', 0))):
            if not isinstance(data, dict):
                continue
            epsilons.append(data.get('epsilon', float(key)))
            validities.append(data.get('validity', 0) * 100)

        if not epsilons:
            plt.close()
            return

        ax.plot(epsilons, validities, 'o-', linewidth=2, markersize=10, color=COLORS['primary'])

        for eps, val in zip(epsilons, validities):
            ax.annotate(f'{val:.1f}%', xy=(eps, val), xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax.set_ylabel('CF Validity (%)', fontsize=12)
        ax.set_title(f'Privacy-Utility Tradeoff - {self.dataset_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'privacy_utility_tradeoff.png'))
        plt.close()

    def plot_fl_aggregation(self):
        """Plot FL aggregation methods comparison"""
        exp4 = self.results['experiments']['exp4_fl_aggregation']
        methods = exp4.get('methods', {})

        if not methods:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        names = []
        accuracies = []
        f1_scores = []

        for name, res in methods.items():
            if not isinstance(res, dict):
                continue
            if 'accuracy' not in res:
                continue
            names.append(name)
            accuracies.append(res.get('accuracy', 0) * 100)
            f1_scores.append(res.get('f1_macro', 0) * 100)

        if not names:
            plt.close()
            return

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color=COLORS['primary'])
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-macro', color=COLORS['secondary'])

        ax.set_ylabel('Score (%)')
        ax.set_title(f'FL Aggregation Methods - {self.dataset_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.set_ylim([0, 105])

        for bar, val in zip(bars1, accuracies):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, f1_scores):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fl_aggregation.png'))
        plt.close()

    def plot_scalability(self):
        """Plot scalability analysis"""
        exp5 = self.results['experiments']['exp5_scalability']
        client_data = exp5.get('clients', {})

        if not client_data:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        clients = []
        accuracies = []
        times = []

        for key, data in sorted(client_data.items(), key=lambda x: int(x[1].get('num_clients', 0))):
            if not isinstance(data, dict):
                continue
            clients.append(data.get('num_clients', int(key)))
            accuracies.append(data.get('accuracy', 0) * 100)
            times.append(data.get('training_time', 0))

        if not clients:
            plt.close()
            return

        # Accuracy vs Clients
        axes[0].plot(clients, accuracies, 'o-', linewidth=2, markersize=10, color=COLORS['primary'])
        axes[0].set_xlabel('Number of Clients')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Accuracy vs Number of Clients', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        for c, a in zip(clients, accuracies):
            axes[0].annotate(f'{a:.1f}%', xy=(c, a), xytext=(5, 5), textcoords='offset points')

        # Time vs Clients
        axes[1].plot(clients, times, 's-', linewidth=2, markersize=10, color=COLORS['secondary'])
        axes[1].set_xlabel('Number of Clients')
        axes[1].set_ylabel('Training Time (seconds)')
        axes[1].set_title('Training Time vs Number of Clients', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        for c, t in zip(clients, times):
            axes[1].annotate(f'{t:.0f}s', xy=(c, t), xytext=(5, 5), textcoords='offset points')

        plt.suptitle(f'Scalability Analysis - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'scalability.png'))
        plt.close()

    def plot_heterogeneity(self):
        """Plot non-IID heterogeneity analysis"""
        exp6 = self.results['experiments']['exp6_heterogeneity']
        alpha_data = exp6.get('alphas', {})

        if not alpha_data:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        alphas = []
        accuracies = []

        for key, data in sorted(alpha_data.items(), key=lambda x: float(x[1].get('alpha', 0) if isinstance(x[1], dict) else 0)):
            if not isinstance(data, dict):
                continue
            alphas.append(data.get('alpha', float(key)))
            accuracies.append(data.get('accuracy', 0) * 100)

        if not alphas:
            plt.close()
            return

        bars = ax.bar([str(a) for a in alphas], accuracies,
                     color=[COLORS['danger'] if a <= 0.1 else COLORS['warning'] if a <= 0.5
                           else COLORS['info'] if a <= 1.0 else COLORS['success'] for a in alphas],
                     edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Dirichlet α (Data Heterogeneity)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Non-IID Data Heterogeneity Impact - {self.dataset_name}', fontweight='bold')

        labels = ['Very High\\nNon-IID' if a <= 0.1 else 'High\\nNon-IID' if a <= 0.5
                  else 'Moderate' if a <= 1.0 else 'Near\\nIID' for a in alphas]

        for bar, val, label in zip(bars, accuracies, labels):
            ax.annotate(f'{val:.1f}%\\n({label})',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)

        ax.set_ylim([0, max(accuracies) * 1.2])

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'heterogeneity.png'))
        plt.close()

    def plot_multiclass_analysis(self):
        """Plot per-class CF analysis"""
        exp7 = self.results['experiments']['exp7_multiclass']
        classes = exp7.get('classes', {})

        if not classes:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        names = []
        validities = []
        proximities = []

        for name, data in classes.items():
            if not isinstance(data, dict):
                continue
            names.append(name)
            validities.append(data.get('validity', 0) * 100)
            proximities.append(data.get('proximity', 0))

        if not names:
            plt.close()
            return

        colors = [COLORS['methods'][i % len(COLORS['methods'])] for i in range(len(names))]

        # Validity per class
        bars1 = axes[0].bar(names, validities, color=colors, edgecolor='black')
        axes[0].set_ylabel('Validity (%)')
        axes[0].set_title('CF Validity per Attack Type', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim([0, 110])

        for bar, val in zip(bars1, validities):
            axes[0].annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)

        # Proximity per class
        bars2 = axes[1].bar(names, proximities, color=colors, edgecolor='black')
        axes[1].set_ylabel('Proximity (L1)')
        axes[1].set_title('CF Proximity per Attack Type', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)

        plt.suptitle(f'Multi-class CF Analysis - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'multiclass_analysis.png'))
        plt.close()

    def plot_ablation(self):
        """Plot ablation study"""
        exp8 = self.results['experiments']['exp8_ablation']
        configs = exp8.get('configs', {})
        contributions = exp8.get('contributions', {})

        if not configs:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        names = []
        validities = []

        for name, data in configs.items():
            if not isinstance(data, dict):
                continue
            names.append(name)
            validities.append(data.get('validity', 0) * 100)

        if not names:
            plt.close()
            return

        colors = [COLORS['primary'] if 'Full' in name else COLORS['warning'] for name in names]

        bars = axes[0].bar(names, validities, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Validity (%)')
        axes[0].set_title('Ablation Configurations', fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, validities):
            axes[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Component contributions
        if contributions:
            comp_names = list(contributions.keys())
            comp_values = [contributions[c] * 100 if isinstance(contributions[c], (int, float)) else 0
                          for c in comp_names]
            colors2 = [COLORS['success'] if v > 0 else COLORS['danger'] for v in comp_values]

            bars2 = axes[1].bar(comp_names, comp_values, color=colors2, edgecolor='black', linewidth=0.5)
            axes[1].set_ylabel('Contribution to Validity (%)')
            axes[1].set_title('Component Contributions (vs Full)', fontweight='bold')
            axes[1].axhline(y=0, color='black', linewidth=0.5)

            for bar, val in zip(bars2, comp_values):
                axes[1].annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')

        plt.suptitle(f'Ablation Study - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'ablation_study.png'))
        plt.close()

    def plot_latency(self):
        """Plot latency analysis"""
        exp9 = self.results['experiments']['exp9_latency']
        latency_data = exp9.get('latency', {})

        if not latency_data:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        iters = []
        mean_times = []
        p95_times = []
        validities = []

        for key, data in sorted(latency_data.items(), key=lambda x: int(x[1].get('max_iter', 0) if isinstance(x[1], dict) else 0)):
            if not isinstance(data, dict):
                continue
            iters.append(data.get('max_iter', int(key)))
            mean_times.append(data.get('mean_time', 0) * 1000)
            p95_times.append(data.get('p95_time', 0) * 1000)
            validities.append(data.get('validity', 0) * 100)

        if not iters:
            plt.close()
            return

        # Time vs Iterations
        axes[0].plot(iters, mean_times, 'o-', linewidth=2, markersize=8,
                    color=COLORS['primary'], label='Mean Time')
        axes[0].plot(iters, p95_times, 's--', linewidth=2, markersize=8,
                    color=COLORS['secondary'], label='P95 Time')
        axes[0].set_xlabel('Max Iterations')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('CF Generation Latency', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Validity vs Iterations
        axes[1].plot(iters, validities, 'o-', linewidth=2, markersize=8, color=COLORS['success'])
        axes[1].set_xlabel('Max Iterations')
        axes[1].set_ylabel('Validity (%)')
        axes[1].set_title('CF Validity vs Iterations', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Latency Analysis - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'latency_analysis.png'))
        plt.close()

    def plot_mia_results(self):
        """Plot MIA attack results"""
        exp10 = self.results['experiments']['exp10_privacy_attack']

        fig, ax = plt.subplots(figsize=(8, 6))

        models = ['No DP', 'With DP']
        base_acc = exp10.get('base_mia_accuracy', 0.5) * 100
        dp_acc = exp10.get('dp_mia_accuracy', 0.5) * 100
        accs = [base_acc, dp_acc]

        colors = [COLORS['danger'], COLORS['success']]
        bars = ax.bar(models, accs, color=colors, edgecolor='black', linewidth=0.5)

        ax.axhline(y=50, color='gray', linestyle='--', label='Random Guess (50%)')
        ax.set_ylabel('MIA Accuracy (%)')
        ax.set_title(f'Membership Inference Attack Results - {self.dataset_name}', fontweight='bold')
        ax.legend()
        ax.set_ylim([40, 60])

        for bar, val in zip(bars, accs):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'mia_results.png'))
        plt.close()

    def plot_cf_feature_importance(self):
        """Plot feature importance from CF case studies"""
        exp11 = self.results['experiments']['exp11_case_study']
        top_features = exp11.get('top_changed_features', [])

        if not top_features:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        names = []
        counts = []
        magnitudes = []

        for i, f in enumerate(top_features[:15]):
            if not isinstance(f, dict):
                continue
            names.append(f.get('feature', f'Feature {i}'))
            counts.append(f.get('change_count', 0))
            magnitudes.append(f.get('avg_magnitude', 0))

        if not names:
            plt.close()
            return

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, counts, width, label='Change Count', color=COLORS['primary'])

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, magnitudes, width, label='Avg Magnitude', color=COLORS['secondary'])

        ax.set_xlabel('Feature')
        ax.set_ylabel('Change Count', color=COLORS['primary'])
        ax2.set_ylabel('Average Magnitude', color=COLORS['secondary'])
        ax.set_title(f'Most Important Features in Counterfactuals - {self.dataset_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cf_feature_importance.png'))
        plt.close()


def generate_figures_from_results(results_path: str, output_dir: str = None):
    """Generate all figures from a results.json file"""
    import json

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(results_path)

    dataset_name = results.get('dataset', 'Unknown')

    visualizer = ExperimentVisualizer(results, output_dir, dataset_name)
    return visualizer.generate_all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        generate_figures_from_results(results_path, output_dir)
    else:
        print("Usage: python visualization.py <results.json> [output_dir]")