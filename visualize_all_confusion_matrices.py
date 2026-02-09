"""
Visualize confusion matrices for all 4 models with all 4 classes.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("white")
plt.rcParams['font.size'] = 10

# Load results
results_path = Path('results_10epochs_freeze_only_vgg/external_test_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# Class names
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Model info
models = [
    ('vgg19', 'VGG19', '#2E86AB'),
    ('resnet50', 'ResNet50', '#A23B72'),
    ('efficientnet_b0', 'EfficientNet-B0', '#F18F01'),
    ('ensemble', 'Ensemble', '#06A77D')
]

# Create figure with 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (model_key, model_name, color) in enumerate(models):
    ax = axes[idx]
    
    # Get confusion matrix
    cm = np.array(results[model_key]['confusion_matrix'])
    
    # Calculate percentages (row-wise normalization)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            cm_percent[i, :] = (cm[i, :] / row_sum) * 100
    
    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Anzahl Predictions'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=cm.max())
    
    # Add custom annotations with both count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            
            # Choose text color based on background
            text_color = 'white' if count > cm.max() * 0.5 else 'black'
            
            # Format text
            if count > 0:
                text = f'{count}\n({percent:.1f}%)'
                fontweight = 'bold' if i == j else 'normal'  # Bold for diagonal
            else:
                text = '0'
                fontweight = 'normal'
            
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color=text_color, fontsize=9,
                   fontweight=fontweight)
    
    # Set title with accuracy
    accuracy = results[model_key]['accuracy'] * 100
    ax.set_title(f'{model_name}\nAccuracy: {accuracy:.2f}%',
                fontsize=13, fontweight='bold', pad=10,
                color=color)
    
    ax.set_ylabel('Wahre Klasse', fontsize=11, fontweight='bold')
    ax.set_xlabel('Vorhergesagte Klasse', fontsize=11, fontweight='bold')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Add overall title
fig.suptitle('Confusion Matrices: Alle 4 Modelle mit allen 4 Klassen\n' + 
             'Externe Validierung (2,004 Glioma + 2,004 Meningioma)',
             fontsize=15, fontweight='bold', y=0.98)

# Add note about no tumor and pituitary
fig.text(0.5, 0.02,
         'Hinweis: No Tumor und Pituitary haben 0 Test-Samples in den externen Daten',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Save
output_path = Path('results_10epochs_freeze_only_vgg/all_confusion_matrices.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Confusion Matrices gespeichert: {output_path}")

# Print summary statistics
print("\n" + "="*70)
print("CONFUSION MATRIX ANALYSE")
print("="*70)

for model_key, model_name, _ in models:
    cm = np.array(results[model_key]['confusion_matrix'])
    accuracy = results[model_key]['accuracy'] * 100
    
    print(f"\n{model_name} (Accuracy: {accuracy:.2f}%):")
    print("-" * 70)
    
    # Only analyze classes with samples (Glioma and Meningioma)
    for i, class_name in enumerate(['Glioma', 'Meningioma']):
        total = cm[i, :].sum()
        if total > 0:
            correct = cm[i, i]
            accuracy_class = (correct / total) * 100
            
            # Find most common misclassification
            misclass_idx = np.argmax([cm[i, j] if j != i else 0 for j in range(4)])
            misclass_count = cm[i, misclass_idx]
            misclass_percent = (misclass_count / total) * 100 if total > 0 else 0
            
            print(f"  {class_name:12s}: {correct:4d}/{total:4d} korrekt ({accuracy_class:5.1f}%)")
            if misclass_count > 0:
                print(f"                Häufigste Fehlklassifikation: {class_names[misclass_idx]} ({misclass_count} = {misclass_percent:.1f}%)")

print("\n" + "="*70)

plt.show()
