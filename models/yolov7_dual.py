import torch
import torch.nn as nn

# --- CBAM Module: Section VI-A ---
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention (Mc) [cite: 134, 135]
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention (Ms) [cite: 134, 135]
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([avg_out, max_out], dim=1))
        return x

# --- BiFPN Layer: Section VI-A-2 ---
class BiFPN_Block(nn.Module):
    """Handles multi-scale tumor features for small necrotic cores [cite: 14, 142]"""
    def __init__(self, channels):
        super(BiFPN_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# --- The Dual-Stream YOLOv7 Model ---
class YOLOv7_DualFusion(nn.Module):
    def __init__(self, num_classes=4): # Glioma, Meningioma, Pituitary, Notumor [cite: 139]
        super(YOLOv7_DualFusion, self).__init__()
        
        # Dual E-ELAN Backbones [cite: 79, 133]
        self.mri_backbone = self._make_backbone()
        self.ct_backbone = self._make_backbone()
        
        # Learnable Fusion Parameter (Alpha) [cite: 114, 115]
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        
        # Attention and Neck
        self.cbam = CBAM(512)
        self.bifpn = BiFPN_Block(512)
        
        # Decoupled Head (Class vs Box) [cite: 95, 119]
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )
        self.classifier = nn.Linear(256, num_classes)
        self.prognostic_head = nn.Linear(256, 1) # Cox-PH [cite: 16, 120]

    def _make_backbone(self):
        # Simplified representation of the E-ELAN / SPD Module [cite: 79, 80]
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True)
        )

    def forward(self, mri, ct):
        # Phase 1: Dual-Stream Processing [cite: 109]
        f_mri = self.mri_backbone(mri)
        f_ct = self.ct_backbone(ct)
        
        # Phase 2: Feature Fusion Nexus [cite: 113, 115]
        f_fused = self.alpha * f_mri + (1 - self.alpha) * f_ct
        
        # Phase 3: Attention & Neck [cite: 81, 142]
        f_refined = self.cbam(f_fused)
        f_bi = self.bifpn(f_refined)
        
        # Phase 4: Clinical Outputs [cite: 104, 118]
        feat = self.detector(f_bi)
        return self.classifier(feat), self.prognostic_head(feat)