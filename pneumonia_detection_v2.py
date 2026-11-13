"""
Pneumonia Detection from Chest X-Rays using Deep Learning
A comprehensive ML pipeline for binary classification of chest X-ray images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Dataset path
    DATA_DIR = r"C:\Users\PC\OneDrive\Desktop\Pneumonia Detection from Chest X-Rays using CNN\chest_xray"
    CLASSES = ["NORMAL", "PNEUMONIA"]
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7
    
    # Image size for model input
    IMG_SIZE = (224, 224)
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model save path
    MODEL_SAVE_PATH = "best_pneumonia_model.pth"
    METRICS_SAVE_PATH = "training_metrics.json"

# ============================================================================
# Seed Function
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

class ChestXRayDataset(Dataset):
    """Custom Dataset for loading chest X-ray images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(data_dir, classes):
    """Load all images from train/val/test subdirectories combined."""
    image_paths = []
    labels = []
    
    # Search in train, val, and test subdirectories
    subdirs = ['train', 'val', 'test']
    
    for subdir in subdirs:
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, subdir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(idx)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Check dataset structure.")
    
    return np.array(image_paths), np.array(labels, dtype=int)

def split_dataset(image_paths, labels, test_size=0.2, val_size=0.2, random_state=42):
    """Split dataset into train, validation, and test sets using stratified split."""
    # First split: separate test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = list(sss1.split(image_paths, labels))[0]
    
    # Second split: separate validation from training
    train_val_labels = labels[train_val_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    train_idx, val_idx = list(sss2.split(train_val_idx, train_val_labels))[0]
    
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    
    return (image_paths[train_idx], labels[train_idx],
            image_paths[val_idx], labels[val_idx],
            image_paths[test_idx], labels[test_idx])

# ============================================================================
# Data Augmentation
# ============================================================================

def get_transforms():
    """Create data augmentation and normalization transforms."""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Validation/Test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_test_transform

# ============================================================================
# Model Architecture
# ============================================================================

class PneumoniaDetectionModel(nn.Module):
    """
    Transfer learning model using ResNet50 backbone with custom head.
    """
    
    def __init__(self, pretrained=True):
        super(PneumoniaDetectionModel, self).__init__()
        
        # Load pretrained ResNet50
        resnet50 = models.resnet50(pretrained=pretrained)
        
        # Modify first layer to accept single-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        
        # ResNet layers
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        
        # Custom classification head
        self.fc = nn.Sequential(
            nn.Linear(resnet50.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({"loss": f"{running_loss/total:.4f}"})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs, device, patience=7):
    """Train the model with early stopping."""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and model checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✓ Best model saved with accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best accuracy: {best_val_acc:.4f}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return all_preds, all_labels, all_probs

def print_metrics(all_preds, all_labels, all_probs, class_names):
    """Print comprehensive evaluation metrics."""
    accuracy = accuracy_score(all_labels, all_preds)
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # ROC-AUC Score
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except:
        print("ROC-AUC Score: N/A")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return accuracy

def plot_results(train_history, all_preds, all_labels, class_names):
    """Plot training curves, confusion matrix, and ROC curve."""
    fig = plt.figure(figsize=(16, 12))
    
    # Training and validation loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_history['train_losses'], label='Train Loss')
    ax1.plot(train_history['val_losses'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(train_history['train_accs'], label='Train Accuracy')
    ax2.plot(train_history['val_accs'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Confusion matrix
    ax3 = plt.subplot(2, 3, 3)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=class_names, yticklabels=class_names)
    ax3.set_title('Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # Distribution of predictions
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist([all_preds[all_labels == 0], all_preds[all_labels == 1]],
             label=class_names, bins=20)
    ax4.set_xlabel('Predicted Class')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Predictions')
    ax4.legend()
    
    # Sample images with predictions
    ax5 = plt.subplot(2, 3, 5)
    correct_count = sum(all_preds == all_labels)
    ax5.text(0.5, 0.5, f'Test Accuracy\n{correct_count}/{len(all_labels)}\n({100*correct_count/len(all_labels):.2f}%)',
             ha='center', va='center', fontsize=16, weight='bold')
    ax5.axis('off')
    
    # Metrics summary
    ax6 = plt.subplot(2, 3, 6)
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    metrics_text = f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    ax6.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Evaluation plots saved as 'evaluation_results.png'")
    plt.show()

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    print("="*60)
    print("PNEUMONIA DETECTION FROM CHEST X-RAYS")
    print("="*60)
    
    # Set seed
    set_seed(Config.RANDOM_SEED)
    print(f"Device: {Config.DEVICE}")
    
    # Load dataset
    print("\n[1] Loading dataset...")
    image_paths, labels = load_dataset(Config.DATA_DIR, Config.CLASSES)
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split dataset
    print("\n[2] Splitting dataset...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        split_dataset(image_paths, labels)
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create datasets and loaders
    print("\n[3] Creating data loaders...")
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = ChestXRayDataset(train_paths, train_labels, train_transform)
    val_dataset = ChestXRayDataset(val_paths, val_labels, val_test_transform)
    test_dataset = ChestXRayDataset(test_paths, test_labels, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n[4] Initializing model...")
    model = PneumoniaDetectionModel(pretrained=True).to(Config.DEVICE)
    print(f"Model loaded on {Config.DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Train model
    print("\n[5] Training model...")
    train_history = train_model(model, train_loader, val_loader, criterion, 
                                optimizer, scheduler, Config.EPOCHS, 
                                Config.DEVICE, patience=Config.EARLY_STOPPING_PATIENCE)
    
    # Evaluate on test set
    print("\n[6] Evaluating on test set...")
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Print metrics and plot results
    print("\n[7] Generating results...")
    accuracy = print_metrics(all_preds, all_labels, all_probs, Config.CLASSES)
    plot_results(train_history, all_preds, all_labels, Config.CLASSES)
    
    # Save metrics
    metrics_summary = {
        'timestamp': datetime.now().isoformat(),
        'test_accuracy': float(accuracy),
        'best_val_accuracy': float(train_history['best_val_acc']),
        'best_epoch': int(train_history['best_epoch']),
        'total_epochs_trained': len(train_history['train_losses']),
        'model_path': Config.MODEL_SAVE_PATH
    }
    
    with open(Config.METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
