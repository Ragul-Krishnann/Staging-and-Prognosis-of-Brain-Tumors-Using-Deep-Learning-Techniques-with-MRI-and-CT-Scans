import torch
import cv2
import numpy as np
from models.yolov7_dual import YOLOv7_DualFusion

def generate_gradcam(model, mri_tensor, ct_tensor, target_layer):
    """
    Implements Grad-CAM as proposed in Paper Section IX-C
    to visualize the model's decision-making process.
    """
    model.eval()
    
    # 1. Register hooks to capture gradients and activations
    activations = []
    gradients = []
    
    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])
    
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    
    # 2. Forward pass
    cls_out, _ = model(mri_tensor, ct_tensor)
    category = torch.argmax(cls_out)
    
    # 3. Backward pass to get gradients
    model.zero_grad()
    cls_out[0, category].backward()
    
    # 4. Compute Grad-CAM Heatmap
    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations[0], dim=1).squeeze().detach().cpu().numpy()
    
    # 5. Post-process: ReLU and Normalization
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    return cam