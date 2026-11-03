"""
Evaluation Script
=================
Loads the trained best model and evaluates it on the test dataset.
Generates metrics (Accuracy, Precision, Recall, F1, AUC-ROC, EER)
and saves Confusion Matrix & ROC curve plots.
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

# Add parent directory to path to support config/src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config
from src.dataset import create_data_loaders
from src.model import build_model

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # 1. Load data loaders
    try:
        _, _, test_loader = create_data_loaders()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure you have run the dataset preparation script first.")
        sys.exit(1)

    # 2. Build model and load checkpoint weights
    model = build_model()
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: No model checkpoint found at {config.BEST_MODEL_PATH}")
        sys.exit(1)

    print(f"Loading best model checkpoint from {config.BEST_MODEL_PATH}...")
    checkpoint = torch.load(config.BEST_MODEL_PATH, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 3. Predict on test loader
    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            probs = torch.sigmoid(model(features)).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    # 4. Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0

    # Equal Error Rate (EER)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    eer = float((fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2)

    print('=' * 50)
    print('  TEST SET RESULTS')
    print('=' * 50)
    print(f'  Accuracy:  {acc:.4f}')
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  EER:       {eer:.4f}")
    print('=' * 50)
    print()
    print(classification_report(all_labels, all_preds, target_names=['Authentic', 'AI-Generated']))

    # 5. Generate and save evaluation plots
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Authentic', 'AI-Generated'],
                yticklabels=['Authentic', 'AI-Generated'], ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix', fontsize=13)

    # ROC Curve
    axes[1].plot(fpr, tpr, color='#00d2ff', linewidth=2, label=f'AUC = {auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plots_path = os.path.join(config.MODELS_DIR, 'evaluation_results.png')
    plt.savefig(plots_path, dpi=150)
    plt.close()
    print(f"Evaluation plots saved to {plots_path}")

if __name__ == "__main__":
    evaluate()
