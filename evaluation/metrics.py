"""
Evaluation and Metrics Module for Brain Tumor Classification

This module provides comprehensive evaluation metrics, model comparison,
and report generation functions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str] = None,
    device: str = None
) -> Dict[str, any]:
    """
    Comprehensive evaluation of a model on test data.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if class_names is None:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    logger.info("Evaluating model...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_probs = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro metrics (average across classes)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ROC-AUC (macro)
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
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }
    
    logger.info(f"Evaluation complete - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc_macro:.4f}")
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: str = None
):
    """
    Plot ROC curves for all classes.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        # One-vs-rest
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = roc_auc_score(y_true_binary, y_score)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {save_path}")
    
    plt.show()


def plot_class_metrics(
    results: Dict[str, any],
    class_names: List[str],
    save_path: str = None
):
    """
    Plot per-class metrics (precision, recall, F1).
    
    Args:
        results: Evaluation results dictionary
        class_names: List of class names
        save_path: Path to save plot
    """
    metrics = {
        'Precision': [results['precision_per_class'][name] for name in class_names],
        'Recall': [results['recall_per_class'][name] for name in class_names],
        'F1-Score': [results['f1_per_class'][name] for name in class_names]
    }
    
    df = pd.DataFrame(metrics, index=class_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.legend(title='Metric', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class metrics plot to {save_path}")
    
    plt.show()


def compare_models(
    models_dict: Dict[str, nn.Module],
    test_loader: DataLoader,
    class_names: List[str] = None,
    device: str = None
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Args:
        models_dict: Dictionary of model_name -> model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run evaluation on
        
    Returns:
        DataFrame with comparison results
    """
    if class_names is None:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    results_list = []
    
    for model_name, model in models_dict.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        results = evaluate_model(model, test_loader, class_names, device)
        
        model_results = {
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision (Macro)': results['precision_macro'],
            'Recall (Macro)': results['recall_macro'],
            'F1-Score (Macro)': results['f1_macro'],
            'ROC-AUC (Macro)': results['roc_auc_macro']
        }
        
        # Add per-class recall (important for tumor detection)
        for class_name in class_names:
            model_results[f'Recall ({class_name})'] = results['recall_per_class'][class_name]
        
        results_list.append(model_results)
    
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.set_index('Model')
    
    # Sort by ROC-AUC
    comparison_df = comparison_df.sort_values('ROC-AUC (Macro)', ascending=False)
    
    logger.info("\nModel Comparison Complete!")
    print("\n" + "=" * 80)
    print(comparison_df.to_string())
    print("=" * 80)
    
    return comparison_df


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = None
):
    """
    Visualize model comparison results.
    
    Args:
        comparison_df: Comparison DataFrame
        save_path: Path to save plot
    """
    # Select main metrics to plot
    main_metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 
                    'F1-Score (Macro)', 'ROC-AUC (Macro)']
    
    plot_df = comparison_df[main_metrics]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    plot_df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Model Comparison - Main Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.legend(title='Metric', fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    
    plt.show()


def generate_comparison_report(
    models_dict: Dict[str, nn.Module],
    test_loader: DataLoader,
    class_names: List[str] = None,
    save_dir: str = "results",
    device: str = None
):
    """
    Generate comprehensive comparison report with all visualizations.
    
    Args:
        models_dict: Dictionary of model_name -> model
        test_loader: Test data loader
        class_names: List of class names
        save_dir: Directory to save results
        device: Device to run evaluation on
    """
    if class_names is None:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("Generating comprehensive comparison report...")
    
    # Compare models
    comparison_df = compare_models(models_dict, test_loader, class_names, device)
    
    # Save comparison table
    comparison_df.to_csv(save_dir / "model_comparison.csv")
    logger.info(f"Saved comparison table to {save_dir / 'model_comparison.csv'}")
    
    # Plot comparison
    plot_model_comparison(comparison_df, save_path=str(save_dir / "model_comparison.png"))
    
    # Evaluate each model individually
    for model_name, model in models_dict.items():
        logger.info(f"\nGenerating detailed report for {model_name}...")
        
        results = evaluate_model(model, test_loader, class_names, device)
        
        # Create model-specific directory
        model_dir = save_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names,
            save_path=str(model_dir / f"{model_name}_confusion_matrix.png"),
            title=f"{model_name} - Confusion Matrix"
        )
        
        # Plot ROC curves
        plot_roc_curves(
            results['y_true'],
            results['y_probs'],
            class_names,
            save_path=str(model_dir / f"{model_name}_roc_curves.png")
        )
        
        # Plot class metrics
        plot_class_metrics(
            results,
            class_names,
            save_path=str(model_dir / f"{model_name}_class_metrics.png")
        )
        
        # Save classification report
        report_df = pd.DataFrame(results['classification_report']).transpose()
        report_df.to_csv(model_dir / f"{model_name}_classification_report.csv")
    
    logger.info(f"\n✓ Comprehensive report generated in {save_dir}")


def predict_with_confidence(
    model: nn.Module,
    image: torch.Tensor,
    class_names: List[str] = None,
    device: str = None
) -> Tuple[str, float, Dict[str, float]]:
    """
    Make prediction with confidence scores for a single image.
    
    Args:
        model: Trained model
        image: Input image tensor
        class_names: List of class names
        device: Device to run on
        
    Returns:
        Tuple of (predicted_class_name, confidence, all_class_probabilities)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if class_names is None:
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    model = model.to(device)
    model.eval()
    
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
        
        predicted_class = predicted.item()
        confidence_value = confidence.item()
        
        # Get all class probabilities
        all_probs = {
            class_names[i]: float(probabilities[0][i])
            for i in range(len(class_names))
        }
    
    return class_names[predicted_class], confidence_value, all_probs


if __name__ == "__main__":
    print("Brain Tumor Classification Evaluation Module")
    print("=" * 50)
    print("\nThis module provides comprehensive evaluation and comparison tools.")
    print("Import and use evaluate_model(), compare_models(), or generate_comparison_report().")
