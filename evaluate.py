"""
=============================================================================
Model Evaluation Script — Run on Google Colab
Generates comprehensive evaluation metrics and visualization plots.
=============================================================================

Usage (Colab):
    1. Run train.py first
    2. Run: python evaluate.py
    3. Output: evaluation_results/ folder with plots and reports

Generates:
    - Classification report (per-class precision, recall, F1)
    - Confusion matrix heatmap
    - Training/validation accuracy curves
    - Training/validation loss curves
    - Per-class accuracy bar chart
    - Weighted & macro metric summaries
    - All saved as PNG images + CSV/JSON files
"""

import os
import json
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    precision_recall_fscore_support
)
from tensorflow.keras.models import load_model

from config import (
    PREPROCESSED_DATA_PATH, MODEL_SAVE_PATH, EVALUATION_PATH,
    create_directories
)


def load_evaluation_data():
    """Load test data, trained model, and label encoder."""
    
    print("[LOADING] Test data...")
    X_test = np.load(os.path.join(PREPROCESSED_DATA_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(PREPROCESSED_DATA_PATH, "y_test.npy"))
    
    print("[LOADING] Label encoder...")
    with open(os.path.join(PREPROCESSED_DATA_PATH, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    
    print("[LOADING] Trained model...")
    model_path = os.path.join(MODEL_SAVE_PATH, "gesture_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_SAVE_PATH, "gesture_model_final.h5")
    model = load_model(model_path)
    
    print("[LOADING] Training history...")
    history_path = os.path.join(MODEL_SAVE_PATH, "training_history.json")
    with open(history_path, "r") as f:
        history = json.load(f)
    
    print(f"[INFO] Test data: X={X_test.shape}, y={y_test.shape}")
    print(f"[INFO] Classes: {len(label_encoder.classes_)}")
    
    return X_test, y_test, model, label_encoder, history


def generate_predictions(model, X_test, y_test):
    """Generate predictions on test data."""
    print("\n[PREDICTING] Running model on test data...")
    
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    return y_pred, y_pred_probs


def plot_confusion_matrix(y_test, y_pred, class_names, save_path):
    """Plot and save confusion matrix heatmap."""
    print("[PLOTTING] Confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Large figure for 56 classes
    fig_size = max(16, len(class_names) * 0.4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix — Gesture Recognition (CNN-LSTM)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_file}")
    
    return cm


def plot_normalized_confusion_matrix(y_test, y_pred, class_names, save_path):
    """Plot and save normalized confusion matrix."""
    print("[PLOTTING] Normalized confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    fig_size = max(16, len(class_names) * 0.4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        square=True,
        vmin=0, vmax=1,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.title('Normalized Confusion Matrix — Gesture Recognition (CNN-LSTM)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "confusion_matrix_normalized.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_file}")


def plot_training_curves(history, save_path):
    """Plot training/validation accuracy and loss curves."""
    print("[PLOTTING] Training curves...")
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Accuracy Curve ---
    axes[0].plot(epochs, history['accuracy'], 'b-o', label='Training Accuracy', 
                 markersize=3, linewidth=2)
    axes[0].plot(epochs, history['val_accuracy'], 'r-s', label='Validation Accuracy', 
                 markersize=3, linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Annotate best validation accuracy
    best_epoch = np.argmax(history['val_accuracy']) + 1
    best_acc = max(history['val_accuracy'])
    axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
    axes[0].annotate(f'Best: {best_acc:.4f}\n(Epoch {best_epoch})',
                     xy=(best_epoch, best_acc),
                     xytext=(best_epoch + 2, best_acc - 0.1),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='green'),
                     color='green')
    
    # --- Loss Curve ---
    axes[1].plot(epochs, history['loss'], 'b-o', label='Training Loss', 
                 markersize=3, linewidth=2)
    axes[1].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', 
                 markersize=3, linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Annotate best validation loss
    best_loss_epoch = np.argmin(history['val_loss']) + 1
    best_loss = min(history['val_loss'])
    axes[1].axvline(x=best_loss_epoch, color='green', linestyle='--', alpha=0.5)
    axes[1].annotate(f'Best: {best_loss:.4f}\n(Epoch {best_loss_epoch})',
                     xy=(best_loss_epoch, best_loss),
                     xytext=(best_loss_epoch + 2, best_loss + 0.2),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='green'),
                     color='green')
    
    plt.suptitle('CNN-LSTM Gesture Recognition — Training History', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "training_curves.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_file}")


def plot_per_class_accuracy(y_test, y_pred, class_names, save_path):
    """Plot per-class accuracy as a horizontal bar chart."""
    print("[PLOTTING] Per-class accuracy...")
    
    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Handle division by zero (classes with no test samples)
    per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)
    
    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_acc = per_class_acc[sorted_indices]
    
    # Color based on accuracy
    colors = ['#e74c3c' if acc < 0.5 else '#f39c12' if acc < 0.75 else '#2ecc71' 
              for acc in sorted_acc]
    
    fig_height = max(10, len(class_names) * 0.35)
    plt.figure(figsize=(12, fig_height))
    
    bars = plt.barh(range(len(sorted_names)), sorted_acc, color=colors, edgecolor='white')
    
    plt.yticks(range(len(sorted_names)), sorted_names, fontsize=9)
    plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy — Gesture Recognition (CNN-LSTM)', 
              fontsize=14, fontweight='bold')
    plt.xlim([0, 1.1])
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy text on bars
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        plt.text(acc + 0.02, i, f'{acc:.2f}', va='center', fontsize=8, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='≥ 75% (Good)'),
        Patch(facecolor='#f39c12', label='50-75% (Fair)'),
        Patch(facecolor='#e74c3c', label='< 50% (Poor)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "per_class_accuracy.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_file}")
    
    return per_class_acc


def plot_per_class_f1(y_test, y_pred, class_names, save_path):
    """Plot per-class F1 score."""
    print("[PLOTTING] Per-class F1 scores...")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_f1 = f1[sorted_indices]
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    
    fig_height = max(10, len(class_names) * 0.4)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    y_pos = np.arange(len(sorted_names))
    bar_width = 0.25
    
    ax.barh(y_pos - bar_width, sorted_precision, bar_width, 
            label='Precision', color='#3498db', alpha=0.8)
    ax.barh(y_pos, sorted_f1, bar_width, 
            label='F1-Score', color='#2ecc71', alpha=0.8)
    ax.barh(y_pos + bar_width, sorted_recall, bar_width, 
            label='Recall', color='#e74c3c', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Precision / Recall / F1-Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim([0, 1.15])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "per_class_f1_scores.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_file}")


def generate_classification_report(y_test, y_pred, class_names, save_path):
    """Generate and save detailed classification report."""
    print("[GENERATING] Classification report...")
    
    # Text report
    report_text = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    
    report_file = os.path.join(save_path, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT — CNN-LSTM Gesture Recognition\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_text)
    print(f"  → Saved: {report_file}")
    
    # Dict report (for CSV/JSON)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    
    report_json = os.path.join(save_path, "classification_report.json")
    with open(report_json, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  → Saved: {report_json}")
    
    # Print to console
    print(f"\n{'='*80}")
    print(report_text)
    print(f"{'='*80}")
    
    return report_dict


def generate_summary_metrics(y_test, y_pred, y_pred_probs, save_path):
    """Generate overall summary metrics."""
    print("[GENERATING] Summary metrics...")
    
    metrics = {
        "overall_accuracy": float(accuracy_score(y_test, y_pred)),
        "weighted_precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        "weighted_recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        "macro_precision": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        "micro_precision": float(precision_score(y_test, y_pred, average='micro', zero_division=0)),
        "micro_recall": float(recall_score(y_test, y_pred, average='micro', zero_division=0)),
        "micro_f1": float(f1_score(y_test, y_pred, average='micro', zero_division=0)),
        "total_test_samples": int(len(y_test)),
        "num_classes": int(len(np.unique(y_test))),
    }
    
    # Save
    metrics_file = os.path.join(save_path, "summary_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → Saved: {metrics_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:     {metrics['overall_accuracy']:.4f}  ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"  ")
    print(f"  Weighted Precision:   {metrics['weighted_precision']:.4f}")
    print(f"  Weighted Recall:      {metrics['weighted_recall']:.4f}")
    print(f"  Weighted F1-Score:    {metrics['weighted_f1']:.4f}")
    print(f"  ")
    print(f"  Macro Precision:      {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:         {metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score:       {metrics['macro_f1']:.4f}")
    print(f"  ")
    print(f"  Total Test Samples:   {metrics['total_test_samples']}")
    print(f"  Number of Classes:    {metrics['num_classes']}")
    print(f"{'='*60}")
    
    return metrics


def plot_learning_rate_history(history, save_path):
    """Plot learning rate changes during training (if available)."""
    if 'lr' in history:
        print("[PLOTTING] Learning rate schedule...")
        
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(history['lr']) + 1), history['lr'], 
                 'g-o', markersize=3, linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_file = os.path.join(save_path, "learning_rate_schedule.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {save_file}")


def plot_prediction_confidence(y_pred_probs, y_test, y_pred, save_path):
    """Plot prediction confidence distribution for correct vs incorrect."""
    print("[PLOTTING] Prediction confidence distribution...")
    
    max_probs = np.max(y_pred_probs, axis=1)
    correct = y_pred == y_test
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(max_probs[correct], bins=30, alpha=0.7, label='Correct Predictions', 
             color='#2ecc71', edgecolor='white')
    plt.hist(max_probs[~correct], bins=30, alpha=0.7, label='Incorrect Predictions', 
             color='#e74c3c', edgecolor='white')
    
    plt.xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "prediction_confidence.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_file}")


# =============================================================================
# MAIN
# =============================================================================
def run_evaluation():
    """Main evaluation pipeline."""
    
    print("=" * 60)
    print("MODEL EVALUATION — Gesture Recognition CNN-LSTM")
    print("=" * 60)
    
    create_directories()
    
    # Load everything
    X_test, y_test, model, label_encoder, history = load_evaluation_data()
    class_names = list(label_encoder.classes_)
    
    # Generate predictions
    y_pred, y_pred_probs = generate_predictions(model, X_test, y_test)
    
    # Generate all metrics and plots
    generate_summary_metrics(y_test, y_pred, y_pred_probs, EVALUATION_PATH)
    generate_classification_report(y_test, y_pred, class_names, EVALUATION_PATH)
    plot_confusion_matrix(y_test, y_pred, class_names, EVALUATION_PATH)
    plot_normalized_confusion_matrix(y_test, y_pred, class_names, EVALUATION_PATH)
    plot_training_curves(history, EVALUATION_PATH)
    plot_per_class_accuracy(y_test, y_pred, class_names, EVALUATION_PATH)
    plot_per_class_f1(y_test, y_pred, class_names, EVALUATION_PATH)
    plot_learning_rate_history(history, EVALUATION_PATH)
    plot_prediction_confidence(y_pred_probs, y_test, y_pred, EVALUATION_PATH)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved to: {EVALUATION_PATH}")
    print(f"\nFiles generated:")
    for f in sorted(os.listdir(EVALUATION_PATH)):
        fpath = os.path.join(EVALUATION_PATH, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  📄 {f} ({size:.1f} KB)")
    
    print(f"\n[DONE] Download the evaluation_results/ folder and share with me!")


if __name__ == "__main__":
    run_evaluation()
