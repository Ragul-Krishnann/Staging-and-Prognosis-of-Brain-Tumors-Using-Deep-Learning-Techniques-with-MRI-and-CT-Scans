import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from models.yolov7_dual import YOLOv7_DualFusion

# --- CONFIGURATION ---
BASE_PATH = r"C:\Users\ragul\Desktop\BrainTumor_Project\data"
MODEL_WEIGHTS = "weights/brain_tumor_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    # 1. Load Data and Model
    test_ds = BrainTumorDataset(BASE_PATH, split='test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    os.makedirs("results", exist_ok=True)
    
    # Initialize Dual-Stream YOLOv7 with Fusion Nexus
    model = YOLOv7_DualFusion(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    print(f"Evaluating {len(test_ds)} images on {DEVICE}...")

    with torch.no_grad():
        for mri, ct, labels in test_loader:
            mri, ct = mri.to(DEVICE), ct.to(DEVICE)
            
            # Forward pass through Fusion Nexus
            cls_out, _ = model(mri, ct)
            
            # Get probabilities for mAP calculation
            probs = torch.softmax(cls_out, dim=1)
            
            _, predicted = torch.max(cls_out, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 2. Precision, Recall, and F1-Score Table
    target_names = ["Glioma", "Meningioma", "Pituitary", "Notumor"]
    print("\n" + "="*45)
    print("--- DETAILED STAGING PERFORMANCE ---")
    print("="*45)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 3. mAP@0.5 Calculation
    # Binarize labels for multi-class mAP
    y_true_binarized = label_binarize(all_labels, classes=[0, 1, 2, 3])
    all_probs_array = np.array(all_probs)
    
    # Calculate Macro Average Precision
    map_score = average_precision_score(y_true_binarized, all_probs_array, average="macro")
    
    print("-" * 45)
    print(f"Overall mAP@0.5 Score: {map_score:.4f}")
    print("-" * 45)

    # 4. Save Final Visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.title(f"Confusion Matrix (mAP: {map_score:.2f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")
    plt.savefig("results/final_evaluation_metrics.png")
    print(f"\n[INFO] Success! All metrics saved to 'results/' folder.")
    plt.show()

if __name__ == "__main__":
    evaluate()