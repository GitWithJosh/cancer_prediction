"""
Training Module for Brain Tumor Classification

This module contains the training pipeline with support for early stopping,
learning rate scheduling, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, value: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            value: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


class Trainer:
    """Trainer class for brain tumor classification models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        device: str = None,
        model_name: str = "model",
        save_dir: str = "checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            model_name: Name for saving checkpoints
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Default criterion and optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        logger.info(f"Trainer initialized for {model_name} on {self.device}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })
        
        val_loss = running_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(
        self,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            lr_scheduler: Learning rate scheduler
            save_best_only: Whether to save only the best model
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            elif not save_best_only:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
            
            # Check early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time / 60:.2f} minutes")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.save_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, path)
            # Also save just the model state dict
            torch.save(self.model.state_dict(), self.save_dir / f"{self.model_name}_best_state.pth")
        else:
            path = self.save_dir / f"{self.model_name}_epoch_{epoch}.pth"
            torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.save_dir / f"{self.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"Saved training history to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    model_name: str = "model",
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 10,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.5,
    device: str = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a model with the specified configuration.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        model_name: Name for saving
        save_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        lr_scheduler_patience: Patience for LR scheduler
        lr_scheduler_factor: Factor for LR reduction
        device: Device to train on
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=lr_scheduler_patience,
        factor=lr_scheduler_factor
    )
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        save_dir=save_dir
    )
    
    # Train
    history = trainer.train(
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler=lr_scheduler,
        save_best_only=True
    )
    
    # Load best model
    best_checkpoint = Path(save_dir) / f"{model_name}_best.pth"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from checkpoint")
    
    return model, history


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None
):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    plt.close()  # Close instead of show to avoid blocking


if __name__ == "__main__":
    print("Brain Tumor Classification Training Module")
    print("=" * 50)
    print("\nThis module provides training utilities for brain tumor classification.")
    print("Import and use the train_model() function or Trainer class for training.")
