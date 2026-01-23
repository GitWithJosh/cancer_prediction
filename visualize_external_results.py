"""
Visualize external test results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# Load results
results_path = Path('results_10epochs_freeze_only_vgg/external_test_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# Create figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
model_colors = {'vgg19': '#2E86AB', 'resnet50': '#A23B72', 'efficientnet_b0': '#F18F01', 'ensemble': '#06A77D'}

# 1. Overall Accuracy Comparison (top left)
ax1 = fig.add_subplot(gs[0, 0])
models = ['VGG19', 'ResNet50', 'EfficientNet-B0', 'Ensemble']
accuracies = [
    results['vgg19']['accuracy'] * 100,
    results['resnet50']['accuracy'] * 100,
    results['efficientnet_b0']['accuracy'] * 100,
    results['ensemble']['accuracy'] * 100
]
bars = ax1.bar(models, accuracies, color=[model_colors[m.lower().replace('-', '_')] for m in models])
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Accuracy auf neuen Test-Bildern', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Random (50%)')
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax1.legend()
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')

# 2. Precision, Recall, F1 Comparison (top middle)
ax2 = fig.add_subplot(gs[0, 1])
metrics = ['Precision', 'Recall', 'F1-Score']
x = np.arange(len(models))
width = 0.25

for i, metric_name in enumerate(metrics):
    metric_key = metric_name.lower().replace('-', '_') + '_macro'
    if metric_name == 'F1-Score':
        metric_key = 'f1_macro'
    values = [results[m.lower().replace('-', '_')][metric_key] * 100 for m in models]
    ax2.bar(x + i*width, values, width, label=metric_name, alpha=0.8)

ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Macro-Averaged Metriken', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim([0, 100])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

# 3. Confusion Matrix - Ensemble (top right)
ax3 = fig.add_subplot(gs[0, 2])
cm_ensemble = np.array(results['ensemble']['confusion_matrix'])
# Only show first 2x2 (glioma, meningioma)
cm_2x2 = cm_ensemble[:2, :2]
sns.heatmap(cm_2x2, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Glioma', 'Meningioma'],
            yticklabels=['Glioma', 'Meningioma'],
            cbar_kws={'label': 'Anzahl'})
ax3.set_title('Ensemble Confusion Matrix\n(Glioma vs Meningioma)', fontsize=14, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

# 4-6. Individual Confusion Matrices
confusion_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
model_names_short = ['vgg19', 'resnet50', 'efficientnet_b0']
model_titles = ['VGG19', 'ResNet50', 'EfficientNet-B0']

for ax, model_name, title in zip(confusion_axes, model_names_short, model_titles):
    cm = np.array(results[model_name]['confusion_matrix'])
    cm_2x2 = cm[:2, :2]
    sns.heatmap(cm_2x2, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=['Glioma', 'Meningioma'],
                yticklabels=['Glioma', 'Meningioma'],
                cbar_kws={'label': 'Anzahl'})
    ax.set_title(f'{title} Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

# 7. Per-Class Performance - Glioma (bottom left)
ax7 = fig.add_subplot(gs[2, 0])
glioma_metrics = []
for model in ['vgg19', 'resnet50', 'efficientnet_b0', 'ensemble']:
    glioma_metrics.append([
        results[model]['precision_per_class']['glioma'] * 100,
        results[model]['recall_per_class']['glioma'] * 100,
        results[model]['f1_per_class']['glioma'] * 100
    ])
glioma_metrics = np.array(glioma_metrics)

x = np.arange(len(models))
width = 0.25
for i, metric in enumerate(['Precision', 'Recall', 'F1']):
    ax7.bar(x + i*width, glioma_metrics[:, i], width, label=metric, alpha=0.8)

ax7.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax7.set_title('Glioma Detection Performance', fontsize=14, fontweight='bold')
ax7.set_xticks(x + width)
ax7.set_xticklabels(models)
ax7.legend()
ax7.set_ylim([0, 100])
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=15, ha='right')

# 8. Per-Class Performance - Meningioma (bottom middle)
ax8 = fig.add_subplot(gs[2, 1])
meningioma_metrics = []
for model in ['vgg19', 'resnet50', 'efficientnet_b0', 'ensemble']:
    meningioma_metrics.append([
        results[model]['precision_per_class']['meningioma'] * 100,
        results[model]['recall_per_class']['meningioma'] * 100,
        results[model]['f1_per_class']['meningioma'] * 100
    ])
meningioma_metrics = np.array(meningioma_metrics)

for i, metric in enumerate(['Precision', 'Recall', 'F1']):
    ax8.bar(x + i*width, meningioma_metrics[:, i], width, label=metric, alpha=0.8)

ax8.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax8.set_title('Meningioma Detection Performance', fontsize=14, fontweight='bold')
ax8.set_xticks(x + width)
ax8.set_xticklabels(models)
ax8.legend()
ax8.set_ylim([0, 100])
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=15, ha='right')

# 9. Summary Table (bottom right)
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('tight')
ax9.axis('off')

summary_data = []
for model, display_name in zip(['vgg19', 'resnet50', 'efficientnet_b0', 'ensemble'], models):
    summary_data.append([
        display_name,
        f"{results[model]['accuracy']*100:.1f}%",
        f"{results[model]['precision_per_class']['glioma']*100:.1f}%",
        f"{results[model]['recall_per_class']['glioma']*100:.1f}%",
        f"{results[model]['precision_per_class']['meningioma']*100:.1f}%",
        f"{results[model]['recall_per_class']['meningioma']*100:.1f}%"
    ])

table = ax9.table(cellText=summary_data,
                  colLabels=['Model', 'Accuracy', 'Gli-Prec', 'Gli-Rec', 'Men-Prec', 'Men-Rec'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax9.set_title('Zusammenfassung der Ergebnisse', fontsize=14, fontweight='bold', pad=20)

# Overall title
fig.suptitle('Externe Test-Bild Evaluierung\n4,008 neue Bilder (2,004 Glioma + 2,004 Meningioma)',
             fontsize=16, fontweight='bold', y=0.995)

# Save
output_path = Path('results_10epochs_freeze_only_vgg/external_test_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualisierung gespeichert: {output_path}")

# Print summary
print("\n" + "="*80)
print("ZUSAMMENFASSUNG DER EXTERNEN TEST-ERGEBNISSE")
print("="*80)
print(f"\nGetestete Bilder: 4,008 (2,004 Glioma + 2,004 Meningioma)")
print("\nModel Performance:")
print("-"*80)
for model, display_name in zip(['vgg19', 'resnet50', 'efficientnet_b0', 'ensemble'], models):
    print(f"\n{display_name}:")
    print(f"  Overall Accuracy:      {results[model]['accuracy']*100:6.2f}%")
    print(f"  Glioma Precision:      {results[model]['precision_per_class']['glioma']*100:6.2f}%")
    print(f"  Glioma Recall:         {results[model]['recall_per_class']['glioma']*100:6.2f}%")
    print(f"  Meningioma Precision:  {results[model]['precision_per_class']['meningioma']*100:6.2f}%")
    print(f"  Meningioma Recall:     {results[model]['recall_per_class']['meningioma']*100:6.2f}%")

print("\n" + "="*80)
print("🏆 BESTES MODELL:")
best_model_idx = np.argmax([results[m]['accuracy'] for m in ['vgg19', 'resnet50', 'efficientnet_b0', 'ensemble']])
best_model = models[best_model_idx]
best_acc = accuracies[best_model_idx]
print(f"  {best_model}: {best_acc:.2f}% Accuracy")
print("="*80)

plt.show()
