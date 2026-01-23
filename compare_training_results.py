"""
Script to compare and visualize results from different training runs.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_results(result_paths):
    """Load all result JSON files."""
    results = {}
    for name, path in result_paths.items():
        with open(path, 'r') as f:
            results[name] = json.load(f)
    return results

def create_comparison_dataframe(results):
    """Create a DataFrame for easy comparison."""
    data = []
    
    for experiment_name, experiment_results in results.items():
        for model_name, metrics in experiment_results.items():
            if model_name == 'ensemble':
                continue  # Handle ensemble separately
            
            row = {
                'Experiment': experiment_name,
                'Model': model_name.upper().replace('_', '-'),
                'Accuracy': metrics['accuracy'] * 100,
                'Precision': metrics['precision_macro'] * 100,
                'Recall': metrics['recall_macro'] * 100,
                'F1-Score': metrics['f1_macro'] * 100,
                'ROC-AUC': metrics['roc_auc_macro'] * 100
            }
            data.append(row)
    
    # Add ensemble results
    for experiment_name, experiment_results in results.items():
        if 'ensemble' in experiment_results:
            metrics = experiment_results['ensemble']
            row = {
                'Experiment': experiment_name,
                'Model': 'ENSEMBLE',
                'Accuracy': metrics['accuracy'] * 100,
                'Precision': metrics['precision_macro'] * 100,
                'Recall': metrics['recall_macro'] * 100,
                'F1-Score': metrics['f1_macro'] * 100,
                'ROC-AUC': metrics['roc_auc_macro'] * 100
            }
            data.append(row)
    
    return pd.DataFrame(data)

def plot_metric_comparison(df, metric, ax, title):
    """Plot comparison for a single metric."""
    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index='Model', columns='Experiment', values=metric)
    
    # Create grouped bar chart
    pivot_df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric} (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.legend(title='Training', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([90, 100])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

def create_overview_plot(df, save_path):
    """Create comprehensive overview plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Vergleich aller Trainings-Durchläufe', fontsize=18, fontweight='bold', y=0.995)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        plot_metric_comparison(df, metric, axes[row, col], metric)
    
    # Remove the last empty subplot
    fig.delaxes(axes[1, 2])
    
    # Add summary table in the last position
    ax_table = fig.add_subplot(2, 3, 6)
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create summary - best results per model
    summary_data = []
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        best_exp = model_df.loc[model_df['Accuracy'].idxmax(), 'Experiment']
        best_acc = model_df['Accuracy'].max()
        summary_data.append([model, best_exp, f"{best_acc:.2f}%"])
    
    table = ax_table.table(cellText=summary_data,
                          colLabels=['Model', 'Best Training', 'Best Accuracy'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.25, 0.45, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax_table.set_title('Beste Ergebnisse pro Modell', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Übersichtsplot gespeichert: {save_path}")
    plt.close()

def create_heatmap(df, save_path):
    """Create heatmap showing all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Heatmap: Vergleich aller Metriken', fontsize=16, fontweight='bold')
    
    experiments = df['Experiment'].unique()
    
    for idx, experiment in enumerate(experiments):
        exp_df = df[df['Experiment'] == experiment]
        
        # Pivot for heatmap
        heatmap_data = exp_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[idx], vmin=90, vmax=100, cbar_kws={'label': 'Score (%)'})
        axes[idx].set_title(experiment, fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap gespeichert: {save_path}")
    plt.close()

def create_detailed_comparison_table(df, save_path):
    """Create detailed comparison table as CSV."""
    # Pivot for better overview
    summary = df.groupby(['Model', 'Experiment']).first().reset_index()
    
    # Save as CSV
    summary.to_csv(save_path, index=False, float_format='%.4f')
    print(f"✅ Detaillierte Tabelle gespeichert: {save_path}")
    
    # Print to console
    print("\n" + "="*100)
    print("DETAILLIERTE ERGEBNISSE")
    print("="*100)
    print(summary.to_string(index=False))
    print("="*100)

def create_line_plot_by_model(df, save_path):
    """Create line plots showing progression across experiments for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Entwicklung der Metriken über verschiedene Trainings', fontsize=16, fontweight='bold')
    
    models = ['VGG19', 'RESNET50', 'EFFICIENTNET-B0', 'ENSEMBLE']
    metrics = ['Accuracy', 'F1-Score', 'Recall', 'ROC-AUC']
    
    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        model_df = df[df['Model'] == model].sort_values('Experiment')
        
        x_pos = range(len(model_df))
        x_labels = model_df['Experiment'].values
        
        for metric in metrics:
            ax.plot(x_pos, model_df[metric].values, marker='o', linewidth=2, label=metric)
        
        ax.set_title(model, fontsize=13, fontweight='bold')
        ax.set_xlabel('Training', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim([90, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Linienplot gespeichert: {save_path}")
    plt.close()

def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*100)
    print("ZUSAMMENFASSUNG")
    print("="*100)
    
    print("\n📊 Durchschnittliche Accuracy pro Experiment:")
    print("-"*50)
    for exp in df['Experiment'].unique():
        avg_acc = df[df['Experiment'] == exp]['Accuracy'].mean()
        print(f"  {exp:40s}: {avg_acc:6.2f}%")
    
    print("\n🏆 Beste Ergebnisse:")
    print("-"*50)
    best_overall = df.loc[df['Accuracy'].idxmax()]
    print(f"  Höchste Accuracy: {best_overall['Model']:20s} in {best_overall['Experiment']:40s} → {best_overall['Accuracy']:.2f}%")
    
    best_recall = df.loc[df['Recall'].idxmax()]
    print(f"  Höchster Recall:  {best_recall['Model']:20s} in {best_recall['Experiment']:40s} → {best_recall['Recall']:.2f}%")
    
    best_f1 = df.loc[df['F1-Score'].idxmax()]
    print(f"  Höchster F1:      {best_f1['Model']:20s} in {best_f1['Experiment']:40s} → {best_f1['F1-Score']:.2f}%")
    
    print("\n📈 Verbesserung durch mehr Epochen (VGG19):")
    print("-"*50)
    vgg_df = df[df['Model'] == 'VGG19'].sort_values('Experiment')
    if len(vgg_df) >= 2:
        for i in range(1, len(vgg_df)):
            prev_acc = vgg_df.iloc[i-1]['Accuracy']
            curr_acc = vgg_df.iloc[i]['Accuracy']
            diff = curr_acc - prev_acc
            print(f"  {vgg_df.iloc[i-1]['Experiment']:40s} → {vgg_df.iloc[i]['Experiment']:40s}: {diff:+.2f}%")
    
    print("="*100 + "\n")

def main():
    # Define result paths
    result_paths = {
        '5 Epochen (nur VGG gefroren)': 'results_5epochs_freeze_only_vgg/all_models_test_results.json',
        '10 Epochen (nur VGG gefroren)': 'results_10epochs_freeze_only_vgg/all_models_test_results.json',
    }
    
    # Check if files exist
    for name, path in result_paths.items():
        if not Path(path).exists():
            print(f"❌ Datei nicht gefunden: {path}")
            return
    
    print("📁 Lade Ergebnisse...")
    results = load_results(result_paths)
    
    print("📊 Erstelle Vergleichs-DataFrame...")
    df = create_comparison_dataframe(results)
    
    # Create output directory
    output_dir = Path('comparison_results_all')
    output_dir.mkdir(exist_ok=True)
    
    print("\n🎨 Erstelle Visualisierungen...")
    
    # Create plots
    create_overview_plot(df, output_dir / 'overview_comparison.png')
    create_heatmap(df, output_dir / 'heatmap_comparison.png')
    create_line_plot_by_model(df, output_dir / 'progression_by_model.png')
    
    # Create table
    create_detailed_comparison_table(df, output_dir / 'detailed_comparison.csv')
    
    # Print summary
    print_summary_statistics(df)
    
    print(f"\n✅ Alle Visualisierungen wurden in '{output_dir}/' gespeichert!")
    print("\nErstellt:")
    print(f"  • overview_comparison.png    - Überblick aller Metriken")
    print(f"  • heatmap_comparison.png     - Heatmap-Vergleich")
    print(f"  • progression_by_model.png   - Entwicklung pro Modell")
    print(f"  • detailed_comparison.csv    - Detaillierte Tabelle")

if __name__ == "__main__":
    main()
