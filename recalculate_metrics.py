"""
Recalculate metrics for external test results - averaging only over the 2 present classes.
"""

import json
from pathlib import Path

# Load original results
results_path = Path('results_10epochs_freeze_only_vgg/external_test_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# Recalculate for each model
for model_name in ['vgg19', 'resnet50', 'efficientnet_b0']:
    model_data = results[model_name]
    
    # Get metrics for glioma and meningioma only
    glioma_precision = model_data['precision_per_class']['glioma']
    meningioma_precision = model_data['precision_per_class']['meningioma']
    
    glioma_recall = model_data['recall_per_class']['glioma']
    meningioma_recall = model_data['recall_per_class']['meningioma']
    
    glioma_f1 = model_data['f1_per_class']['glioma']
    meningioma_f1 = model_data['f1_per_class']['meningioma']
    
    # Calculate 2-class averages
    model_data['precision_macro_2class'] = (glioma_precision + meningioma_precision) / 2
    model_data['recall_macro_2class'] = (glioma_recall + meningioma_recall) / 2
    model_data['f1_macro_2class'] = (glioma_f1 + meningioma_f1) / 2
    
    # Keep old 4-class averages for reference
    model_data['precision_macro_4class'] = model_data['precision_macro']
    model_data['recall_macro_4class'] = model_data['recall_macro']
    model_data['f1_macro_4class'] = model_data['f1_macro']
    
    # Replace main macro values with 2-class versions
    model_data['precision_macro'] = model_data['precision_macro_2class']
    model_data['recall_macro'] = model_data['recall_macro_2class']
    model_data['f1_macro'] = model_data['f1_macro_2class']
    
    print(f"\n{model_name.upper()}:")
    print(f"  Precision: {model_data['precision_macro_4class']*100:.2f}% (4-class) → {model_data['precision_macro_2class']*100:.2f}% (2-class)")
    print(f"  Recall:    {model_data['recall_macro_4class']*100:.2f}% (4-class) → {model_data['recall_macro_2class']*100:.2f}% (2-class)")
    print(f"  F1-Score:  {model_data['f1_macro_4class']*100:.2f}% (4-class) → {model_data['f1_macro_2class']*100:.2f}% (2-class)")

# For ensemble, calculate from confusion matrix
ensemble_data = results['ensemble']
cm = ensemble_data['confusion_matrix']

# Calculate precision and recall for glioma and meningioma
glioma_tp = cm[0][0]
glioma_fp = cm[1][0]  # meningioma predicted as glioma
glioma_fn = cm[0][1]  # glioma predicted as meningioma
glioma_precision = glioma_tp / (glioma_tp + glioma_fp) if (glioma_tp + glioma_fp) > 0 else 0
glioma_recall = glioma_tp / (glioma_tp + glioma_fn + cm[0][2] + cm[0][3]) if (glioma_tp + glioma_fn + cm[0][2] + cm[0][3]) > 0 else 0
glioma_f1 = 2 * (glioma_precision * glioma_recall) / (glioma_precision + glioma_recall) if (glioma_precision + glioma_recall) > 0 else 0

meningioma_tp = cm[1][1]
meningioma_fp = cm[0][1]  # glioma predicted as meningioma
meningioma_fn = cm[1][0]  # meningioma predicted as glioma
meningioma_precision = meningioma_tp / (meningioma_tp + meningioma_fp) if (meningioma_tp + meningioma_fp) > 0 else 0
meningioma_recall = meningioma_tp / (meningioma_tp + meningioma_fn + cm[1][2] + cm[1][3]) if (meningioma_tp + meningioma_fn + cm[1][2] + cm[1][3]) > 0 else 0
meningioma_f1 = 2 * (meningioma_precision * meningioma_recall) / (meningioma_precision + meningioma_recall) if (meningioma_precision + meningioma_recall) > 0 else 0

# Add per-class metrics for ensemble
ensemble_data['precision_per_class'] = {
    'glioma': glioma_precision,
    'meningioma': meningioma_precision,
    'notumor': 0.0,
    'pituitary': 0.0
}
ensemble_data['recall_per_class'] = {
    'glioma': glioma_recall,
    'meningioma': meningioma_recall,
    'notumor': 0.0,
    'pituitary': 0.0
}
ensemble_data['f1_per_class'] = {
    'glioma': glioma_f1,
    'meningioma': meningioma_f1,
    'notumor': 0.0,
    'pituitary': 0.0
}

# Calculate 2-class averages for ensemble
ensemble_data['precision_macro_2class'] = (glioma_precision + meningioma_precision) / 2
ensemble_data['recall_macro_2class'] = (glioma_recall + meningioma_recall) / 2
ensemble_data['f1_macro_2class'] = (glioma_f1 + meningioma_f1) / 2

# Keep old 4-class averages
ensemble_data['precision_macro_4class'] = ensemble_data['precision_macro']
ensemble_data['recall_macro_4class'] = ensemble_data['recall_macro']
ensemble_data['f1_macro_4class'] = ensemble_data['f1_macro']

# Replace with 2-class versions
ensemble_data['precision_macro'] = ensemble_data['precision_macro_2class']
ensemble_data['recall_macro'] = ensemble_data['recall_macro_2class']
ensemble_data['f1_macro'] = ensemble_data['f1_macro_2class']

print(f"\nENSEMBLE:")
print(f"  Precision: {ensemble_data['precision_macro_4class']*100:.2f}% (4-class) → {ensemble_data['precision_macro_2class']*100:.2f}% (2-class)")
print(f"  Recall:    {ensemble_data['recall_macro_4class']*100:.2f}% (4-class) → {ensemble_data['recall_macro_2class']*100:.2f}% (2-class)")
print(f"  F1-Score:  {ensemble_data['f1_macro_4class']*100:.2f}% (4-class) → {ensemble_data['f1_macro_2class']*100:.2f}% (2-class)")

# Save updated results
output_path = Path('results_10epochs_freeze_only_vgg/external_test_results_corrected.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Korrigierte Metriken gespeichert: {output_path}")
print("\nHinweis: Die Macro-Averages wurden nun nur über Glioma und Meningioma berechnet.")
print("Die alten 4-Klassen-Werte sind als *_macro_4class gespeichert.")
