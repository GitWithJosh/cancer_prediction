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
plt.rcParams['font.size'] = 10

# Load results
results_path = Path('results_10epochs_freeze_only_vgg/external_test_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# Create figure - kompakter (2x2 layout)
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

# Color palette
model_colors = {'vgg19': '#2E86AB', 'resnet50': '#A23B72', 'efficientnet_b0': '#F18F01', 'ensemble': '#06A77D'}

# 1. Overall Accuracy & F1-Score (top left)
ax1 = fig.add_subplot(gs[0, 0])
models = ['VGG19', 'ResNet50', 'EfficientNet-B0', 'Ensemble']
model_keys = ['vgg19', 'resnet50', 'efficientnet_b0', 'ensemble']
accuracies = [results[m]['accuracy'] * 100 for m in model_keys]
f1_scores = [results[m]['f1_macro'] * 100 for m in model_keys]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', 
                color=[model_colors[m] for m in model_keys], alpha=0.8)
bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1-Score (Macro)',
                color=[model_colors[m] for m in model_keys], alpha=0.5)

ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance (4,008 externe Test-Bilder)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=20, ha='right')
ax1.set_ylim([0, 100])
ax1.legend(loc='lower right')
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.2, linewidth=1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# 2. Ensemble Confusion Matrix (top right)
ax2 = fig.add_subplot(gs[0, 1])
cm_ensemble = np.array(results['ensemble']['confusion_matrix'])
cm_2x2 = cm_ensemble[:2, :2]

# Normalize for better visualization
cm_norm = cm_2x2.astype('float') / cm_2x2.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_2x2, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Glioma', 'Meningioma'],
            yticklabels=['Glioma', 'Meningioma'],
            cbar_kws={'label': 'Anzahl'}, linewidths=1, linecolor='white')

# Add percentages
for i in range(2):
    for j in range(2):
        text = ax2.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                       ha='center', va='center', fontsize=8, color='darkblue', style='italic')

ax2.set_title('Ensemble Confusion Matrix\nAccuracy: 91.32%', fontsize=12, fontweight='bold')
ax2.set_ylabel('Wahre Klasse', fontsize=11, fontweight='bold')
ax2.set_xlabel('Vorhergesagte Klasse', fontsize=11, fontweight='bold')

# 3. Glioma vs Meningioma Detection Performance (bottom left)
ax3 = fig.add_subplot(gs[1, 0])

# Calculate metrics for ensemble
cm_ens = np.array(results['ensemble']['confusion_matrix'])
gli_prec = (cm_ens[0,0] / (cm_ens[0,0] + cm_ens[1,0])) * 100
gli_rec = (cm_ens[0,0] / cm_ens[0,:].sum()) * 100
men_prec = (cm_ens[1,1] / (cm_ens[0,1] + cm_ens[1,1])) * 100
men_rec = (cm_ens[1,1] / cm_ens[1,:].sum()) * 100

# Bar chart
metrics = ['Precision', 'Recall']
glioma_vals = [gli_prec, gli_rec]
meningioma_vals = [men_prec, men_rec]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, glioma_vals, width, label='Glioma', color='#E63946', alpha=0.8)
bars2 = ax3.bar(x + width/2, meningioma_vals, width, label='Meningioma', color='#457B9D', alpha=0.8)

ax3.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax3.set_title('Ensemble: Klassen-spezifische Metriken', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.set_ylim([0, 100])
ax3.legend()
ax3.axhline(y=90, color='green', linestyle='--', alpha=0.3, linewidth=1, label='90% Ziel')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Performance Summary Table (bottom right)
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('tight')
ax4.axis('off')

summary_data = []
for model, display_name in zip(model_keys, models):
    if 'precision_per_class' in results[model]:
        gli_f1 = results[model]['f1_per_class']['glioma'] * 100
        men_f1 = results[model]['f1_per_class']['meningioma'] * 100
    else:
        cm = np.array(results[model]['confusion_matrix'])
        gli_prec = (cm[0,0] / (cm[0,0] + cm[1,0])) * 100 if (cm[0,0] + cm[1,0]) > 0 else 0
        gli_rec = (cm[0,0] / cm[0,:].sum()) * 100 if cm[0,:].sum() > 0 else 0
        men_prec = (cm[1,1] / (cm[0,1] + cm[1,1])) * 100 if (cm[0,1] + cm[1,1]) > 0 else 0
        men_rec = (cm[1,1] / cm[1,:].sum()) * 100 if cm[1,:].sum() > 0 else 0
        gli_f1 = 2 * (gli_prec * gli_rec) / (gli_prec + gli_rec) if (gli_prec + gli_rec) > 0 else 0
        men_f1 = 2 * (men_prec * men_rec) / (men_prec + men_rec) if (men_prec + men_rec) > 0 else 0
    
    summary_data.append([
        display_name,
        f"{results[model]['accuracy']*100:.1f}%",
        f"{gli_f1:.1f}%",
        f"{men_f1:.1f}%"
    ])

table = ax4.table(cellText=summary_data,
                  colLabels=['Model', 'Accuracy', 'Glioma F1', 'Menin. F1'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.30, 0.23, 0.23, 0.23])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.8)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model (ensemble)
for j in range(4):
    table[(4, j)].set_facecolor('#D4EDDA')
    table[(4, j)].set_text_props(weight='bold')

# Alternate row colors
for i in range(1, 4):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F8F9FA')

ax4.set_title('Vergleich aller Modelle', fontsize=12, fontweight='bold', pad=20)

# Overall title
fig.suptitle('Externe Validierung: Glioma & Meningioma Klassifikation\n2,004 Glioma + 2,004 Meningioma Bilder',
             fontsize=13, fontweight='bold', y=0.98)

# Save
output_path = Path('results_10epochs_freeze_only_vgg/external_test_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Visualisierung gespeichert: {output_path}")

# Print summary
print("\n" + "="*70)
print("EXTERNE VALIDIERUNG - ZUSAMMENFASSUNG")
print("="*70)
print(f"Testdaten: 4,008 Bilder (2,004 Glioma + 2,004 Meningioma)")
print("\n🏆 ENSEMBLE MODEL (Best Performance):")
print(f"  • Overall Accuracy:      {results['ensemble']['accuracy']*100:6.2f}%")
cm_ens = np.array(results['ensemble']['confusion_matrix'])
gli_prec = (cm_ens[0,0] / (cm_ens[0,0] + cm_ens[1,0])) * 100
gli_rec = (cm_ens[0,0] / cm_ens[0,:].sum()) * 100
men_prec = (cm_ens[1,1] / (cm_ens[0,1] + cm_ens[1,1])) * 100
men_rec = (cm_ens[1,1] / cm_ens[1,:].sum()) * 100
print(f"  • Glioma:      Precision {gli_prec:5.1f}%  |  Recall {gli_rec:5.1f}%")
print(f"  • Meningioma:  Precision {men_prec:5.1f}%  |  Recall {men_rec:5.1f}%")
print("\nIndividuelle Modelle:")
print(f"  • EfficientNet-B0:  {results['efficientnet_b0']['accuracy']*100:5.1f}%")
print(f"  • VGG19:            {results['vgg19']['accuracy']*100:5.1f}%")
print(f"  • ResNet50:         {results['resnet50']['accuracy']*100:5.1f}%")
print("="*70)

plt.show()
