import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Initial Conv -> Downsampling -> 9 ResBlocks -> Upsampling
        model = [nn.ReflectionPad2d(3), nn.Conv2d(1, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True)]
        # Downsampling
        for i in range(2):
            model += [nn.Conv2d(64 * (2**i), 64 * (2**(i+1)), 3, stride=2, padding=1),
                      nn.InstanceNorm2d(64 * (2**(i+1))), nn.ReLU(True)]
        # Residual blocks (preserving tumor anatomy)
        for _ in range(9): model += [ResidualBlock(256)]
        # Upsampling
        for i in range(2):
            model += [nn.ConvTranspose2d(256 // (2**i), 256 // (2**(i+1)), 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(256 // (2**(i+1))), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)