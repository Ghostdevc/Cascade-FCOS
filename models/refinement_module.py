import torch
import torch.nn as nn
import torchvision.ops as ops

class LightweightFCM(nn.Module):
    """
    Feature Consistency Module (FCM) using an inverted bottleneck and Deformable Conv.
    Ensures feature alignment across cascade stages with minimal inference overhead.
    """
    def __init__(self, in_channels=256, mid_channels=64):
        super().__init__()
        
        # 1. Dimensionality Reduction (1x1 Conv)
        self.reduce_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 2. Spatial Offset Predictor for the DCN
        # A 3x3 kernel has 9 locations. Each needs a (dx, dy) offset, so 18 channels.
        self.offset_conv = nn.Conv2d(mid_channels, 2 * 3 * 3, kernel_size=3, padding=1)
        
        # Initialize offsets to ZERO. 
        # This is critical! It ensures the DCN starts acting as a standard 3x3 conv 
        # at the beginning of training, preventing early instability.
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # 3. Deformable Convolution for Feature Alignment
        self.deform_conv = ops.DeformConv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.deform_bn = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 4. Dimensionality Expansion (1x1 Conv)
        self.expand_conv = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        identity = x
        
        # Step 1: Compress channels (256 -> 64)
        out = self.relu1(self.reduce_bn(self.reduce_conv(x)))
        
        # Step 2: Predict spatial deformations
        offsets = self.offset_conv(out)
        
        # Step 3: Warp features via Deformable Convolution
        out = self.relu2(self.deform_bn(self.deform_conv(out, offsets)))
        
        # Step 4: Expand channels back (64 -> 256)
        out = self.expand_bn(self.expand_conv(out))
        
        # Step 5: Residual connection
        return torch.relu(identity + out)

# --- Usage Example ---
# If we have an FPN feature map P4:
# fcm_module = LightweightFCM()
# dummy_p4 = torch.randn(2, 256, 100, 134) # Batch, Channels, H, W
# aligned_p4 = fcm_module(dummy_p4)
# print("Input shape:", dummy_p4.shape)
# print("Aligned shape:", aligned_p4.shape)