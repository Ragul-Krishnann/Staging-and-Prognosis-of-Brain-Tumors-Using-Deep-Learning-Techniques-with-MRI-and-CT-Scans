import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from glob import glob

class BrainTumorDataset(Dataset):
    """
    Implements Layer 1: Data Preparation [cite: 69]
    Standardizes intensity via Z-score normalization[cite: 97].
    """
    def __init__(self, base_path, split='train', transform=None):
        self.base_path = os.path.normpath(base_path)
        
        # Pattern to find MRI images: data/mri/train/category/*.jpg
        search_pattern = os.path.join(self.base_path, 'mri', split, '*', '*')
        self.mri_paths = sorted(glob(search_pattern))
        
        self.transform = transform
        self.label_map = {
            "glioma": 0,
            "meningioma": 1,
            "pituitary": 2,
            "notumor": 3 
        }

        if len(self.mri_paths) == 0:
            raise ValueError(f"No images found in {search_pattern}. Check your paths!")

    def __len__(self):
        """Returns the total number of samples [cite: 102]"""
        return len(self.mri_paths)

    def __getitem__(self, idx):
        """
        REQUIRED METHOD: Retrieves a paired MRI and Synthetic CT sample.
        """
        # 1. Setup Paths
        mri_path = os.path.normpath(self.mri_paths[idx])
        ct_path = mri_path.replace(os.sep + 'mri' + os.sep, os.sep + 'ct' + os.sep)
        
        # 2. Extract Label
        class_name = mri_path.split(os.sep)[-2].lower()
        label = self.label_map.get(class_name, 3)

        # 3. Read and Resize Images
        # Standardizing to 256x256 is vital for the BiFPN neck [cite: 141, 142]
        mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
        ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)

        if mri_img is None: mri_img = np.zeros((256, 256), dtype=np.uint8)
        if ct_img is None: ct_img = np.zeros((256, 256), dtype=np.uint8)

        mri_img = cv2.resize(mri_img, (256, 256), interpolation=cv2.INTER_AREA)
        ct_img = cv2.resize(ct_img, (256, 256), interpolation=cv2.INTER_AREA)

        # 4. Z-score Normalization 
        # Formula: z = (x - mean) / std
        mri_img = mri_img.astype(np.float32)
        ct_img = ct_img.astype(np.float32)
        
        mri_img = (mri_img - np.mean(mri_img)) / (np.std(mri_img) + 1e-8)
        ct_img = (ct_img - np.mean(ct_img)) / (np.std(ct_img) + 1e-8)

        # 5. Convert to PyTorch Tensors
        mri_tensor = torch.from_numpy(mri_img).unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_img).unsqueeze(0)

        return mri_tensor, ct_tensor, torch.tensor(label, dtype=torch.long)