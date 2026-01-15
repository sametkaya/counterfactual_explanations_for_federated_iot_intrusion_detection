"""
FedCF-IoT: Visualization Module
===============================
Generates all figures for the experiments in publication-quality format.

Figures Generated:
1. Confusion Matrix Heatmap
2. Training Curve (Loss/Accuracy vs Epoch)
3. CF Methods Comparison (Bar Chart)
4. Privacy-Utility Tradeoff Curve
5. Scalability Analysis (Clients vs Accuracy/Time)
6. Non-IID Heterogeneity (Alpha vs Accuracy)
7. ROC Curves (Per-class)
8. Feature Importance from CFs
9. Baseline Model Comparison
10. Latency vs Validity Tradeoff
11. Class Distribution Pie Chart
12. Ablation Study Bar Chart
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import os

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Matplotlib >= 3.6
except:
    try:
        plt.style.use('seaborn-whitegrid')  # Matplotlib < 3.6
    except:
        pass  # Use default style
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

# Color palette for consistency
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
        """
        Args:
            results: Dictionary containing all experiment results
            output_dir: Directory to save figures
            dataset_name: Name of dataset for figure titles
        """
        self.results = results
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def generate_all(self):
        """Generate all figures"""
        print("\n" + "="*60)
        print("GENERATING FIGURES")
        print("="*60)
        
        generated = []
        
        # 1. Class Distribution
        if 'dataset_info' in self.results:
            self.plot_class_distribution()
            generated.append("class_distribution.png")
        
        # 2. Training Curve
        if 'exp1_training' in self.results.get('experiments', {}):
            self.plot_training_curve()
            generated.append("training_curve.png")
            
            # 3. Confusion Matrix
            if 'classification_metrics' in self.results['experiments']['exp1_training']:
                self.plot_confusion_matrix()
                generated.append("confusion_matrix.png")
                
                # ROC Curves
                self.plot_roc_curves()
                generated.append("roc_curves.png")
            
            # 4. Baseline Comparison
            if 'baseline_comparison' in self.results['experiments']['exp1_training']:
                self.plot_baseline_comparison()
                generated.append("baseline_comparison.png")
        
        # 5. CF Methods Comparison
        if 'exp2_cf_comparison' in self.results.get('experiments', {}):
            self.plot_cf_comparison()
            generated.append("cf_methods_comparison.png")
            
            self.plot_cf_comparison_detailed()
            generated.append("cf_methods_detailed.png")
        
        # 6. Privacy-Utility Tradeoff
        if 'exp3_privacy' in self.results.get('experiments', {}):
            self.plot_privacy_utility()
            generated.append("privacy_utility_tradeoff.png")
        
        # 7. FL Aggregation
        if 'exp4_fl_aggregation' in self.results.get('experiments', {}):
            self.plot_fl_aggregation()
            generated.append("fl_aggregation.png")
        
        # 8. Scalability
        if 'exp5_scalability' in self.results.get('experiments', {}):
            self.plot_scalability()
            generated.append("scalability.png")
        
        # 9. Non-IID Heterogeneity
        if 'exp6_heterogeneity' in self.results.get('experiments', {}):
            self.plot_heterogeneity()
            generated.append("heterogeneity.png")
        
        # 10. Multi-class Analysis
        if 'exp7_multiclass' in self.results.get('experiments', {}):
            self.plot_multiclass_analysis()
            generated.append("multiclass_analysis.png")
        
        # 11. Ablation Study
        if 'exp8_ablation' in self.results.get('experiments', {}):
            self.plot_ablation()
            generated.append("ablation_study.png")
        
        # 12. Latency Analysis
        if 'exp9_latency' in self.results.get('experiments', {}):
            self.plot_latency()
            generated.append("latency_analysis.png")
        
        # 13. Privacy Attack (MIA)
        if 'exp10_privacy_attack' in self.results.get('experiments', {}):
            self.plot_mia_results()
            generated.append("mia_results.png")
        
        # 14. Case Study Feature Importance
        if 'exp11_case_study' in self.results.get('experiments', {}):
            self.plot_cf_feature_importance()
            generated.append("cf_feature_importance.png")
        
        print(f"\nGenerated {len(generated)} figures in {self.figures_dir}/")
        for fig in generated:
            print(f"  - {fig}")
        
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
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = axes[0].pie(
            values, labels=None, autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
            colors=colors, startangle=90
        )
        axes[0].set_title('Class Distribution', fontweight='bold')
        axes[0].legend(wedges, labels, title="Classes", loc="center left", 
                      bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Bar chart
        bars = axes[1].bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Sample Count per Class', fontweight='bold')
        axes[1].set_yscale('log')
        
        # Add value labels on bars
        for bar, val, pct in zip(bars, values, percentages):
            axes[1].annotate(f'{val:,}\n({pct:.1f}%)', 
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'class_distribution.png'))
        plt.close()
        print("  [✓] class_distribution.png")
    
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
        
        # Add final accuracy annotation
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
        print("  [✓] training_curve.png")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        exp1 = self.results['experiments']['exp1_training']
        clf_metrics = exp1.get('classification_metrics', {})
        cm = clf_metrics.get('confusion_matrix', [])
        
        if not cm:
            return
        
        cm = np.array(cm)
        
        # Get class names
        per_class = clf_metrics.get('per_class_metrics', {})
        class_names = list(per_class.keys()) if per_class else [f'Class {i}' for i in range(len(cm))]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Proportion'})
        
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(f'Normalized Confusion Matrix - {self.dataset_name}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'))
        plt.close()
        print("  [✓] confusion_matrix.png")
    
    def plot_roc_curves(self):
        """Plot ROC curves placeholder (requires probabilities)"""
        exp1 = self.results['experiments']['exp1_training']
        clf_metrics = exp1.get('classification_metrics', {})
        per_class = clf_metrics.get('per_class_metrics', {})
        
        if not per_class:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Since we don't have full ROC data, plot per-class performance
        class_names = list(per_class.keys())
        f1_scores = [per_class[c]['f1'] for c in class_names]
        recall_scores = [per_class[c]['recall'] for c in class_names]
        precision_scores = [per_class[c]['precision'] for c in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', color=COLORS['primary'])
        bars2 = ax.bar(x, recall_scores, width, label='Recall', color=COLORS['secondary'])
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color=COLORS['success'])
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Performance Metrics - {self.dataset_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.axhline(y=clf_metrics.get('f1_macro', 0), color='red', linestyle='--', 
                   label=f"Macro F1: {clf_metrics.get('f1_macro', 0):.3f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'roc_curves.png'))
        plt.close()
        print("  [✓] roc_curves.png (per-class metrics)")
    
    def plot_baseline_comparison(self):
        """Plot baseline model comparison"""
        exp1 = self.results['experiments']['exp1_training']
        baselines = exp1.get('baseline_comparison', {})
        clf_metrics = exp1.get('classification_metrics', {})
        
        if not baselines:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        models = ['MLP (Ours)'] + list(baselines.keys())
        accuracies = [clf_metrics.get('accuracy', 0)]
        f1_macros = [clf_metrics.get('f1_macro', 0)]
        
        for name, res in baselines.items():
            accuracies.append(res.get('accuracy', 0))
            f1_macros.append(res.get('f1_macro', 0))
        
        x = np.arange(len(models))
        colors = [COLORS['primary']] + [COLORS['methods'][i % len(COLORS['methods'])] 
                                         for i in range(len(baselines))]
        
        # Accuracy comparison
        bars1 = axes[0].bar(x, accuracies, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylim([0, 1.05])
        
        # Add value labels
        for bar, val in zip(bars1, accuracies):
            axes[0].annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10)
        
        # F1-Score comparison
        bars2 = axes[1].bar(x, f1_macros, color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('F1-Score (Macro)')
        axes[1].set_title('F1-Score Comparison', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylim([0, 1.05])
        
        for bar, val in zip(bars2, f1_macros):
            axes[1].annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f'Model Comparison - {self.dataset_name}', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'baseline_comparison.png'))
        plt.close()
        print("  [✓] baseline_comparison.png")
    
    def plot_cf_comparison(self):
        """Plot CF methods comparison bar chart"""
        exp2 = self.results['experiments']['exp2_cf_comparison']
        methods = exp2.get('methods', {})
        
        if not methods:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        names = []
        validities = []
        errors = []
        
        for name, data in methods.items():
            if isinstance(data, dict) and 'validity' in data:
                names.append(name.replace('_', '\n'))
                validities.append(data['validity'] * 100)
                ci = data.get('validity_ci', (data['validity'], data['validity']))
                err = (data['validity'] - ci[0]) * 100
                errors.append(err)
        
        x = np.arange(len(names))
        colors = [COLORS['methods'][i % len(COLORS['methods'])] for i in range(len(names))]
        
        # Highlight our method
        for i, name in enumerate(names):
            if 'FedCF' in name and 'DP' in name:
                colors[i] = COLORS['success']
        
        bars = ax.bar(x, validities, yerr=errors, capsize=5, color=colors, 
                     edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel('Validity (%)')
        ax.set_title(f'Counterfactual Methods Comparison - {self.dataset_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim([0, 110])
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, val in zip(bars, validities):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 2),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cf_methods_comparison.png'))
        plt.close()
        print("  [✓] cf_methods_comparison.png")
    
    def plot_cf_comparison_detailed(self):
        """Plot detailed CF comparison with multiple metrics"""
        exp2 = self.results['experiments']['exp2_cf_comparison']
        methods = exp2.get('methods', {})
        
        if not methods:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = []
        validities = []
        proximities = []
        sparsities = []
        
        for name, data in methods.items():
            if isinstance(data, dict) and 'validity' in data:
                names.append(name)
                validities.append(data['validity'] * 100)
                proximities.append(data.get('proximity', 0))
                sparsities.append(data.get('sparsity', 0))
        
        x = np.arange(len(names))
        colors = [COLORS['methods'][i % len(COLORS['methods'])] for i in range(len(names))]
        
        # Validity
        axes[0, 0].bar(x, validities, color=colors, edgecolor='black', linewidth=0.5)
        axes[0, 0].set_ylabel('Validity (%)')
        axes[0, 0].set_title('Validity', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        # Proximity
        axes[0, 1].bar(x, proximities, color=colors, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_ylabel('Proximity (L1)')
        axes[0, 1].set_title('Proximity (Lower is Better)', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        # Sparsity
        axes[1, 0].bar(x, sparsities, color=colors, edgecolor='black', linewidth=0.5)
        axes[1, 0].set_ylabel('Sparsity (# Changes)')
        axes[1, 0].set_title('Sparsity (Lower is Better)', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        axes[1, 0].axhline(y=10, color='red', linestyle='--', label='Constraint: max 10')
        axes[1, 0].legend()
        
        # Summary radar/table
        axes[1, 1].axis('off')
        summary_text = "Method Performance Summary\n" + "="*40 + "\n\n"
        summary_text += f"{'Method':<25} {'Valid':>8} {'Prox':>8} {'Spars':>8}\n"
        summary_text += "-"*50 + "\n"
        for i, name in enumerate(names[:8]):  # Top 8
            summary_text += f"{name:<25} {validities[i]:>7.1f}% {proximities[i]:>8.4f} {sparsities[i]:>8.1f}\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Detailed CF Methods Comparison - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cf_methods_detailed.png'))
        plt.close()
        print("  [✓] cf_methods_detailed.png")
    
    def plot_privacy_utility(self):
        """Plot privacy-utility tradeoff curve"""
        exp3 = self.results['experiments']['exp3_privacy']
        epsilons_data = exp3.get('epsilons', {})
        
        if not epsilons_data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epsilons = []
        validities = []
        ci_lower = []
        ci_upper = []
        
        for key, data in sorted(epsilons_data.items(), key=lambda x: float(x[1].get('epsilon', 0))):
            epsilons.append(data.get('epsilon', float(key)))
            validities.append(data.get('validity', 0) * 100)
            ci = data.get('validity_ci', (data.get('validity', 0), data.get('validity', 0)))
            ci_lower.append(ci[0] * 100)
            ci_upper.append(ci[1] * 100)
        
        # Plot with confidence interval
        ax.fill_between(epsilons, ci_lower, ci_upper, alpha=0.3, color=COLORS['primary'])
        ax.plot(epsilons, validities, 'o-', linewidth=2, markersize=10, 
                color=COLORS['primary'], label='CF Validity')
        
        # Annotations
        for eps, val in zip(epsilons, validities):
            ax.annotate(f'{val:.1f}%', xy=(eps, val), xytext=(5, 5),
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax.set_ylabel('CF Validity (%)', fontsize=12)
        ax.set_title(f'Privacy-Utility Tradeoff - {self.dataset_name}', fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim([90, 102])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add privacy level annotations
        ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Strong Privacy')
        ax.axvline(x=5.0, color='orange', linestyle='--', alpha=0.5, label='Moderate Privacy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'privacy_utility_tradeoff.png'))
        plt.close()
        print("  [✓] privacy_utility_tradeoff.png")
    
    def plot_fl_aggregation(self):
        """Plot FL aggregation methods comparison"""
        exp4 = self.results['experiments']['exp4_fl_aggregation']
        methods = exp4.get('methods', {})
        
        if not methods:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(methods.keys())
        accuracies = [methods[n].get('accuracy', 0) * 100 for n in names]
        
        colors = [COLORS['primary'] if n == 'Centralized' else COLORS['methods'][i % len(COLORS['methods'])] 
                  for i, n in enumerate(names)]
        
        bars = ax.bar(names, accuracies, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'FL Aggregation Methods Comparison - {self.dataset_name}', fontweight='bold')
        ax.set_ylim([min(accuracies) * 0.95, max(accuracies) * 1.02])
        
        # Add baseline line
        cent_acc = methods.get('Centralized', {}).get('accuracy', 0) * 100
        ax.axhline(y=cent_acc, color='red', linestyle='--', alpha=0.7, 
                   label=f'Centralized: {cent_acc:.2f}%')
        ax.legend()
        
        # Value labels
        for bar, val in zip(bars, accuracies):
            ax.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fl_aggregation.png'))
        plt.close()
        print("  [✓] fl_aggregation.png")
    
    def plot_scalability(self):
        """Plot scalability analysis - clients vs accuracy and time"""
        exp5 = self.results['experiments']['exp5_scalability']
        client_data = exp5.get('client_counts', {})
        
        if not client_data:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        clients = []
        accuracies = []
        times = []
        
        for key, data in sorted(client_data.items(), key=lambda x: int(x[1].get('num_clients', 0))):
            clients.append(data.get('num_clients', int(key)))
            accuracies.append(data.get('accuracy', 0) * 100)
            times.append(data.get('train_time', 0))
        
        # Accuracy vs Clients
        axes[0].plot(clients, accuracies, 'o-', linewidth=2, markersize=10, color=COLORS['primary'])
        axes[0].set_xlabel('Number of Clients')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Accuracy vs Number of Clients', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        for c, a in zip(clients, accuracies):
            axes[0].annotate(f'{a:.2f}%', xy=(c, a), xytext=(5, 5), textcoords='offset points')
        
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
        print("  [✓] scalability.png")
    
    def plot_heterogeneity(self):
        """Plot non-IID heterogeneity analysis"""
        exp6 = self.results['experiments']['exp6_heterogeneity']
        alpha_data = exp6.get('alphas', {})
        
        if not alpha_data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        alphas = []
        accuracies = []
        
        for key, data in sorted(alpha_data.items(), key=lambda x: float(x[1].get('alpha', 0))):
            alphas.append(data.get('alpha', float(key)))
            accuracies.append(data.get('accuracy', 0) * 100)
        
        bars = ax.bar([str(a) for a in alphas], accuracies, 
                     color=[COLORS['danger'] if a <= 0.1 else COLORS['warning'] if a <= 0.5 
                           else COLORS['info'] if a <= 1.0 else COLORS['success'] for a in alphas],
                     edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Dirichlet α (Data Heterogeneity)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Non-IID Data Heterogeneity Impact - {self.dataset_name}', fontweight='bold')
        
        # Add heterogeneity level labels
        labels = ['Very High\nNon-IID' if a <= 0.1 else 'High\nNon-IID' if a <= 0.5 
                  else 'Moderate' if a <= 1.0 else 'Near\nIID' for a in alphas]
        
        for bar, val, label in zip(bars, accuracies, labels):
            ax.annotate(f'{val:.2f}%\n({label})', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim([0, max(accuracies) * 1.15])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'heterogeneity.png'))
        plt.close()
        print("  [✓] heterogeneity.png")
    
    def plot_multiclass_analysis(self):
        """Plot per-class CF analysis"""
        exp7 = self.results['experiments']['exp7_multiclass']
        classes = exp7.get('classes', {})
        
        if not classes:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        names = list(classes.keys())
        validities = [classes[n].get('validity', 0) * 100 for n in names]
        proximities = [classes[n].get('proximity', 0) for n in names]
        samples = [classes[n].get('n_samples', 0) for n in names]
        
        colors = [COLORS['methods'][i % len(COLORS['methods'])] for i in range(len(names))]
        
        # Validity per class
        bars1 = axes[0].bar(names, validities, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Validity (%)')
        axes[0].set_title('CF Validity per Attack Type', fontweight='bold')
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        axes[0].set_ylim([0, 110])
        
        for bar, val, n in zip(bars1, validities, samples):
            axes[0].annotate(f'{val:.1f}%\n(n={n})', 
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        # Proximity per class
        bars2 = axes[1].bar(names, proximities, color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('Proximity (L1)')
        axes[1].set_title('CF Proximity per Attack Type', fontweight='bold')
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        
        for bar, val in zip(bars2, proximities):
            axes[1].annotate(f'{val:.4f}', 
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Multi-Class CF Analysis - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'multiclass_analysis.png'))
        plt.close()
        print("  [✓] multiclass_analysis.png")
    
    def plot_ablation(self):
        """Plot ablation study results"""
        exp8 = self.results['experiments']['exp8_ablation']
        configs = exp8.get('configs', {})
        contributions = exp8.get('contributions', {})
        
        if not configs:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Configuration comparison
        names = list(configs.keys())
        validities = [configs[n].get('validity', 0) * 100 for n in names]
        
        colors = [COLORS['success'] if n == 'Full' else COLORS['methods'][i % len(COLORS['methods'])] 
                  for i, n in enumerate(names)]
        
        bars = axes[0].bar(names, validities, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Validity (%)')
        axes[0].set_title('Ablation Study: Component Removal', fontweight='bold')
        axes[0].set_ylim([0, 110])
        
        for bar, val in zip(bars, validities):
            axes[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Component contributions
        if contributions:
            comp_names = list(contributions.keys())
            comp_values = [contributions[c] * 100 for c in comp_names]
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
        print("  [✓] ablation_study.png")
    
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
        
        for key, data in sorted(latency_data.items(), key=lambda x: int(x[1].get('max_iter', 0))):
            iters.append(data.get('max_iter', int(key)))
            mean_times.append(data.get('mean_time', 0) * 1000)  # Convert to ms
            p95_times.append(data.get('p95_time', 0) * 1000)
            validities.append(data.get('validity', 0) * 100)
        
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
        axes[1].set_ylim([0, 110])
        
        for it, val in zip(iters, validities):
            axes[1].annotate(f'{val:.1f}%', xy=(it, val), xytext=(5, 5), textcoords='offset points')
        
        plt.suptitle(f'Latency Analysis - {self.dataset_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'latency_analysis.png'))
        plt.close()
        print("  [✓] latency_analysis.png")
    
    def plot_mia_results(self):
        """Plot MIA (Membership Inference Attack) results"""
        exp10 = self.results['experiments']['exp10_privacy_attack']
        mia_results = exp10.get('mia_results', {})
        
        if not mia_results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        models = []
        threshold_acc = []
        loss_acc = []
        
        for model_type, data in mia_results.items():
            if isinstance(data, dict) and 'threshold_mia' in data:
                models.append('No DP' if model_type == 'no_dp' else 'With DP')
                threshold_acc.append(data['threshold_mia'].get('attack_accuracy', 0.5) * 100)
                loss_acc.append(data['loss_mia'].get('attack_accuracy', 0.5) * 100)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, threshold_acc, width, label='Threshold MIA', color=COLORS['primary'])
        bars2 = ax.bar(x + width/2, loss_acc, width, label='Loss-based MIA', color=COLORS['secondary'])
        
        ax.axhline(y=50, color='red', linestyle='--', label='Random Guess (50%)')
        
        ax.set_ylabel('Attack Accuracy (%)')
        ax.set_title(f'Membership Inference Attack Results - {self.dataset_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([45, 60])
        
        # Value labels
        for bar, val in zip(bars1, threshold_acc):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)
        for bar, val in zip(bars2, loss_acc):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'mia_results.png'))
        plt.close()
        print("  [✓] mia_results.png")
    
    def plot_cf_feature_importance(self):
        """Plot feature importance from CF case studies"""
        exp11 = self.results['experiments']['exp11_case_study']
        top_features = exp11.get('top_changed_features', [])
        
        if not top_features:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = [f.get('feature', f'Feature {i}') for i, f in enumerate(top_features[:15])]
        counts = [f.get('change_count', 0) for f in top_features[:15]]
        magnitudes = [f.get('avg_magnitude', 0) for f in top_features[:15]]
        
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
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cf_feature_importance.png'))
        plt.close()
        print("  [✓] cf_feature_importance.png")


def generate_figures_from_results(results_path: str, output_dir: str = None):
    """
    Generate all figures from a results.json file
    
    Args:
        results_path: Path to results.json
        output_dir: Output directory (default: same as results file)
    """
    import json
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    
    dataset_name = results.get('dataset', 'Unknown')
    
    visualizer = ExperimentVisualizer(results, output_dir, dataset_name)
    visualizer.generate_all()
    
    return visualizer.figures_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        generate_figures_from_results(results_path, output_dir)
    else:
        print("Usage: python visualization.py <results.json> [output_dir]")
