"""
Test trained models on external/new test images.

Usage:
    python evaluation/test_external_images.py --data_dir path/to/test/images
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import yaml
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging

from models.architectures import BrainTumorClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalTestDataset(Dataset):
    """Dataset for external test images."""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing subdirectories with images
            transform: Transformations to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {
            'glioma': 0,
            'meningioma': 1,
            'notumor': 2,
            'pituitary': 3
        }
        
        # Map folder names to class names
        folder_mapping = {
            'brain_glioma': 'glioma',
            'Menin': 'meningioma',
            'meningioma': 'meningioma',
            'notumor': 'notumor',
            'no_tumor': 'notumor',
            'pituitary': 'pituitary'
        }
        
        # Scan directories
        for folder in self.data_dir.iterdir():
            if not folder.is_dir():
                continue
            
            # Map folder name to class
            folder_name = folder.name.lower()
            class_name = None
            for key, value in folder_mapping.items():
                if key.lower() in folder_name:
                    class_name = value
                    break
            
            if class_name is None:
                logger.warning(f"Unknown folder: {folder.name}, skipping...")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Load images
            for img_path in folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), class_idx))
        
        logger.info(f"Loaded {len(self.samples)} images from {data_dir}")
        
        # Print class distribution
        class_counts = {}
        for _, label in self.samples:
            class_name = list(self.class_to_idx.keys())[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_model(checkpoint_path, model_name, device):
    """Load a trained model from checkpoint."""
    model = BrainTumorClassifier(
        model_name=model_name,
        num_classes=4,
        pretrained=False
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_name} from {checkpoint_path}")
    return model


def evaluate_model(model, test_loader, class_names, device):
    """Evaluate model on test set."""
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_per_class': {class_names[i]: float(precision_per_class[i]) for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: float(recall_per_class[i]) for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: float(f1_per_class[i]) for i in range(len(class_names))},
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def ensemble_predict(models, test_loader, device):
    """Make ensemble predictions."""
    all_predictions = []
    all_labels = []
    
    for model in models.values():
        model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Ensemble Testing'):
            images = images.to(device)
            
            # Get predictions from each model
            model_preds = []
            for model in models.values():
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = probabilities.max(1)
                model_preds.append(predictions.cpu().numpy())
            
            # Majority voting
            model_preds = np.array(model_preds)  # (num_models, batch_size)
            
            for i in range(images.size(0)):
                votes = model_preds[:, i]
                # Count votes
                unique, counts = np.unique(votes, return_counts=True)
                ensemble_pred = unique[np.argmax(counts)]
                all_predictions.append(ensemble_pred)
            
            all_labels.extend(labels.numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Test models on external images')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test images (with subdirs: brain_glioma, Menin, etc.)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_10epochs_freeze_only_vgg',
                       help='Directory containing model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--output_file', type=str, default='external_test_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("EXTERNAL TEST IMAGE EVALUATION")
    logger.info("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    class_names = config['dataset']['class_names']
    
    # Setup transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['preprocessing']['normalize_mean'],
            std=config['preprocessing']['normalize_std']
        )
    ])
    
    # Load dataset
    logger.info(f"\nLoading images from: {args.data_dir}")
    dataset = ExternalTestDataset(args.data_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load models
    checkpoint_dir = Path(args.checkpoint_dir)
    models = {}
    
    for model_name in ['vgg19', 'resnet50', 'efficientnet_b0']:
        checkpoint_path = checkpoint_dir / f'{model_name}_brain_tumor_best_state.pth'
        if checkpoint_path.exists():
            models[model_name] = load_model(checkpoint_path, model_name, device)
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    if not models:
        logger.error("No models loaded! Check checkpoint directory.")
        return
    
    # Test individual models
    results = {}
    
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL MODEL RESULTS")
    logger.info("="*60)
    
    for model_name, model in models.items():
        logger.info(f"\nTesting {model_name.upper()}...")
        model_results = evaluate_model(model, test_loader, class_names, device)
        results[model_name] = model_results
        
        logger.info(f"\n{model_name.upper()} Results:")
        logger.info(f"  Accuracy:  {model_results['accuracy']:.4f}")
        logger.info(f"  Precision: {model_results['precision_macro']:.4f}")
        logger.info(f"  Recall:    {model_results['recall_macro']:.4f}")
        logger.info(f"  F1-Score:  {model_results['f1_macro']:.4f}")
        
        logger.info(f"\nClassification Report:")
        logger.info(f"\n{model_results['classification_report']}")
    
    # Test ensemble
    if len(models) >= 2:
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE RESULTS")
        logger.info("="*60)
        
        ensemble_preds, true_labels = ensemble_predict(models, test_loader, device)
        
        accuracy = accuracy_score(true_labels, ensemble_preds)
        precision = precision_score(true_labels, ensemble_preds, average='macro', zero_division=0)
        recall = recall_score(true_labels, ensemble_preds, average='macro', zero_division=0)
        f1 = f1_score(true_labels, ensemble_preds, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, ensemble_preds)
        report = classification_report(true_labels, ensemble_preds, target_names=class_names)
        
        results['ensemble'] = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info(f"\nEnsemble Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        logger.info(f"\nClassification Report:")
        logger.info(f"\n{report}")
    
    # Save results
    output_path = Path('results_10epochs_freeze_only_vgg') / args.output_file
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"\n✅ Results saved to: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
