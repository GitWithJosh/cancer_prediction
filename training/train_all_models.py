"""
Training script to train all models sequentially.

Usage:
    python training/train_all_models.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import json
from data.preprocessing import load_and_split_dataset, create_data_loaders
from models.architectures import create_model
from training.trainer import train_model, plot_training_history
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    Path(config['paths']['results_dir']).mkdir(exist_ok=True, parents=True)
    Path(config['paths']['checkpoints_dir']).mkdir(exist_ok=True, parents=True)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load and split dataset
    logger.info("Loading dataset...")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(
        data_path=config['dataset']['data_path'],
        test_size=config['dataset']['test_size'],
        val_size=config['dataset']['val_size'],
        image_size=config['dataset']['image_size'],
        seed=config['seed']
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4
    )
    # Models to train
    model_names = ['vgg19', 'resnet50', 'efficientnet_b0']
    
    # Import evaluation functions
    from evaluation.metrics import evaluate_model
    
    # Store all results
    all_results = {}
    
    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}\n")
        
        # Create model
        model = create_model(
            model_name=model_name,
            num_classes=config['dataset']['num_classes'],
            pretrained=config['models'][model_name]['pretrained'],
            freeze_features=config['models'][model_name].get('freeze_features', False),
            device=device
        )
        
        # Train model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['epochs'],
            lr=config['training']['learning_rate'],
            weight_decay=0.0001,
            model_name=f"{model_name}_brain_tumor",
            save_dir=config['paths']['checkpoints_dir'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            lr_scheduler_patience=config['training']['lr_scheduler_patience'],
            lr_scheduler_factor=config['training']['lr_scheduler_factor'],
            device=device
        )
        
        # Plot training history
        plot_training_history(
            history,
            save_path=Path(config['paths']['results_dir']) / f"{model_name}_training_history.png"
        )
        
        # Evaluate on test set
        logger.info(f"\nEvaluating {model_name.upper()} on test set...")
        test_results = evaluate_model(
            trained_model,
            test_loader,
            class_names=config['dataset']['class_names'],
            device=device
        )
        
        # Log test results
        logger.info(f"\n{model_name.upper()} Test Results:")
        logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"  ROC-AUC: {test_results['roc_auc_macro']:.4f}")
        logger.info(f"  Precision: {test_results['precision_macro']:.4f}")
        logger.info(f"  Recall: {test_results['recall_macro']:.4f}")
        logger.info(f"  F1-Score: {test_results['f1_macro']:.4f}")
        
        # Save test results to dictionary
        all_results[model_name] = {
            'accuracy': float(test_results['accuracy']),
            'roc_auc_macro': float(test_results['roc_auc_macro']),
            'precision_macro': float(test_results['precision_macro']),
            'recall_macro': float(test_results['recall_macro']),
            'f1_macro': float(test_results['f1_macro']),
            'precision_per_class': {k: float(v) for k, v in test_results['precision_per_class'].items()},
            'recall_per_class': {k: float(v) for k, v in test_results['recall_per_class'].items()},
            'f1_per_class': {k: float(v) for k, v in test_results['f1_per_class'].items()}
        }
        
        logger.info(f"\n✓ {model_name.upper()} training completed\n")
    
    # Save all results to JSON
    results_path = Path(config['paths']['results_dir']) / 'all_models_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"\n{'='*60}")
    logger.info("All models trained successfully!")
    logger.info(f"Test results saved to: {results_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
