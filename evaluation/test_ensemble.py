"""
Test Ensemble Model on Test Dataset

Combines predictions from VGG19, ResNet50, and EfficientNet-B0.
Uses highest confidence when all 3 models predict different classes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import logging
import json
import sys
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.architectures import BrainTumorClassifier
from data.preprocessing import load_and_split_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensemble_predict(models, inputs, device):
    """
    Ensemble prediction with confidence-based voting.
    
    Strategy:
    1. Get predictions from all 3 models
    2. If all agree → use that prediction
    3. If 2 agree → use majority vote
    4. If all disagree → use prediction with highest confidence
    
    Args:
        models: Dictionary of model_name -> model
        inputs: Input batch
        device: Device to run on
        
    Returns:
        predictions: Array of predicted classes
        confidences: Array of confidence scores
    """
    batch_size = inputs.size(0)
    all_predictions = []
    all_confidences = []
    all_probabilities = []
    
    # Get predictions from each model
    for model_name, model in models.items():
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    # Convert to arrays: shape (num_models, batch_size)
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    all_probabilities = np.array(all_probabilities)  # (num_models, batch_size, num_classes)
    
    # Ensemble decision for each sample
    ensemble_preds = []
    ensemble_confs = []
    
    for i in range(batch_size):
        preds = all_predictions[:, i]  # Predictions from all models for sample i
        confs = all_confidences[:, i]  # Confidences from all models for sample i
        
        # Count votes for each class
        unique, counts = np.unique(preds, return_counts=True)
        
        if counts.max() >= 2:
            # Majority vote (at least 2 models agree)
            majority_class = unique[counts.argmax()]
            # Use average confidence of models that voted for majority class
            majority_mask = preds == majority_class
            avg_confidence = confs[majority_mask].mean()
            
            ensemble_preds.append(majority_class)
            ensemble_confs.append(avg_confidence)
        else:
            # All models disagree - use highest confidence
            max_conf_idx = confs.argmax()
            ensemble_preds.append(preds[max_conf_idx])
            ensemble_confs.append(confs[max_conf_idx])
    
    return np.array(ensemble_preds), np.array(ensemble_confs), all_probabilities


def evaluate_ensemble(models, test_loader, class_names, device):
    """
    Evaluate ensemble model on test set.
    
    Args:
        models: Dictionary of trained models
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating ensemble model on test set...")
    
    # Set all models to eval mode
    for model in models.values():
        model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing Ensemble'):
            inputs = inputs.to(device)
            
            # Get ensemble predictions
            preds, confs, probs = ensemble_predict(models, inputs, device)
            
            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confs)
            all_probabilities.append(probs)
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_conf = np.array(all_confidences)
    
    # Average probabilities across models for ROC-AUC
    # Shape: (num_models, num_batches, batch_size, num_classes)
    all_probs_array = np.concatenate([p.transpose(1, 0, 2) for p in all_probabilities], axis=0)
    # Average across models: (num_samples, num_classes)
    y_probs = all_probs_array.mean(axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ROC-AUC
    try:
        roc_auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except:
        roc_auc_macro = 0.0
        logger.warning("Could not calculate ROC-AUC score")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_auc_macro': roc_auc_macro,
        'precision_per_class': {class_names[i]: precision_per_class[i] for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: recall_per_class[i] for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: f1_per_class[i] for i in range(len(class_names))},
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'average_confidence': float(y_conf.mean())
    }
    
    return results


def main():
    """Main function to test ensemble model."""
    
    logger.info("="*60)
    logger.info("ENSEMBLE MODEL EVALUATION")
    logger.info("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("\nLoading test dataset...")
    _, _, test_dataset = load_and_split_dataset(
        data_path=config['dataset']['data_path'],
        val_size=config['dataset']['val_size'],
        test_size=config['dataset']['test_size'],
        seed=config['seed']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0  # Use 0 for compatibility
    )
    
    logger.info(f"Test set size: {len(test_dataset)} images")
    
    # Load models
    logger.info("\nLoading trained models...")
    models = {}
    model_names = ['vgg19', 'resnet50', 'efficientnet_b0']
    checkpoints_dir = Path('checkpoints_10epochs_freeze_none')
    
    for model_name in model_names:
        checkpoint_path = checkpoints_dir / f'{model_name}_brain_tumor_best_state.pth'
        
        if checkpoint_path.exists():
            logger.info(f"  Loading {model_name}...")
            model = BrainTumorClassifier(
                model_name=model_name,
                num_classes=config['dataset']['num_classes'],
                pretrained=False
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model = model.to(device)
            model.eval()
            models[model_name] = model
        else:
            logger.error(f"  Checkpoint not found: {checkpoint_path}")
            logger.error(f"  Please train {model_name} first!")
            return
    
    if len(models) != 3:
        logger.error(f"\nError: Need all 3 models for ensemble. Found {len(models)}")
        return
    
    logger.info(f"\n✓ Successfully loaded {len(models)} models")
    
    # Evaluate ensemble
    results = evaluate_ensemble(
        models,
        test_loader,
        config['dataset']['class_names'],
        device
    )
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE TEST RESULTS")
    logger.info("="*60)
    logger.info(f"\n⭐ Accuracy:           {results['accuracy']:.4f} ({results['accuracy']:.2%})")
    logger.info(f"⭐ Recall (Macro):     {results['recall_macro']:.4f} ({results['recall_macro']:.2%})")
    logger.info(f"   Precision (Macro):  {results['precision_macro']:.4f} ({results['precision_macro']:.2%})")
    logger.info(f"   F1-Score (Macro):   {results['f1_macro']:.4f} ({results['f1_macro']:.2%})")
    logger.info(f"   ROC-AUC (Macro):    {results['roc_auc_macro']:.4f} ({results['roc_auc_macro']:.2%})")
    logger.info(f"   Avg Confidence:     {results['average_confidence']:.4f} ({results['average_confidence']:.2%})")
    
    logger.info("\n" + "-"*60)
    logger.info("Per-Class Recall:")
    logger.info("-"*60)
    for class_name in config['dataset']['class_names']:
        recall = results['recall_per_class'][class_name]
        precision = results['precision_per_class'][class_name]
        f1 = results['f1_per_class'][class_name]
        logger.info(f"  {class_name:12s}: Recall={recall:.2%}  Precision={precision:.2%}  F1={f1:.2%}")
    
    # Save results
    results_dir = Path('results_10epochs_freeze_none')
    results_dir.mkdir(exist_ok=True)
    
    # Load existing results if available
    results_file = results_dir / 'all_models_test_results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    # Add ensemble results
    all_results['ensemble'] = {
        'accuracy': float(results['accuracy']),
        'roc_auc_macro': float(results['roc_auc_macro']),
        'precision_macro': float(results['precision_macro']),
        'recall_macro': float(results['recall_macro']),
        'f1_macro': float(results['f1_macro']),
        'precision_per_class': {k: float(v) for k, v in results['precision_per_class'].items()},
        'recall_per_class': {k: float(v) for k, v in results['recall_per_class'].items()},
        'f1_per_class': {k: float(v) for k, v in results['f1_per_class'].items()},
        'average_confidence': float(results['average_confidence'])
    }
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"\n✓ Results saved to {results_file}")
    
    # Save confusion matrix
    cm_file = results_dir / 'ensemble_confusion_matrix.txt'
    with open(cm_file, 'w') as f:
        f.write("Ensemble Confusion Matrix\n")
        f.write("="*60 + "\n\n")
        f.write("            " + "  ".join([f"{c:12s}" for c in config['dataset']['class_names']]) + "\n")
        for i, row in enumerate(results['confusion_matrix']):
            f.write(f"{config['dataset']['class_names'][i]:12s}" + "  ".join([f"{val:12d}" for val in row]) + "\n")
    
    logger.info(f"✓ Confusion matrix saved to {cm_file}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Ensemble evaluation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
