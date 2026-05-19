import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class FCOSBackboneFPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        
        # 1. Load Pretrained ResNet-50
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Extract necessary layers
        # C3: stride 8, channels 512
        self.c3_extractor = nn.Sequential(
            backbone.conv1, 
            backbone.bn1, 
            backbone.relu, 
            backbone.maxpool, 
            backbone.layer1, 
            backbone.layer2
        )
        # C4: stride 16, channels 1024
        self.c4_extractor = backbone.layer3
        
        # C5: stride 32, channels 2048
        self.c5_extractor = backbone.layer4
        
        # Freeze BatchNorm layers (standard practice for object detection to stabilize training)
        self._freeze_bn()

        # 2. Lateral Connections (1x1 convolutions to unify channel dimensions)
        self.lat_prj3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lat_prj4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lat_prj5 = nn.Conv2d(2048, out_channels, kernel_size=1)

        # 3. Top-Down Smoothing Convolutions (3x3 convolutions to mitigate upsampling aliasing)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 4. P6 and P7 Convolutions (Derived progressively as per FCOS paper)
        self.p6_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        # Initialize FPN specific weights
        self._init_fpn_weights()

    def _freeze_bn(self):
        """Freezes running means and variances in BatchNorm layers."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                    
    def _init_fpn_weights(self):
        """Kaiming Uniform initialization for FPN layers."""
        for m in [self.lat_prj3, self.lat_prj4, self.lat_prj5, 
                  self.smooth3, self.smooth4, self.smooth5, 
                  self.p6_conv, self.p7_conv]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # --- Bottom-up pathway ---
        c3 = self.c3_extractor(x)
        c4 = self.c4_extractor(c3)
        c5 = self.c5_extractor(c4)

        # --- Lateral connections ---
        p5_lat = self.lat_prj5(c5)
        p4_lat = self.lat_prj4(c4)
        p3_lat = self.lat_prj3(c3)

        # --- Top-down pathway ---
        p5 = p5_lat
        p4 = p4_lat + F.interpolate(p5, size=p4_lat.shape[-2:], mode='nearest')
        p3 = p3_lat + F.interpolate(p4, size=p3_lat.shape[-2:], mode='nearest')

        # --- Smoothing ---
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        # --- P6 and P7 ---
        p6 = self.p6_conv(p5)
        p7 = self.p7_conv(F.relu(p6))

        return [p3, p4, p5, p6, p7]

# --- Usage Example ---
# fpn_model = FCOSBackboneFPN()
# dummy_img = torch.randn(2, 3, 800, 1066) # Example padded batch shape
# features = fpn_model(dummy_img)
# for f in features:
#     print(f.shape) 
# Expected output spatial dimensions will be approx: /8, /16, /32, /64, /128



class StageSpecificFCOSHead(nn.Module):
    """
    The FCOS Head for a single Cascade Stage. 
    In Cascade FCOS, we instantiate one of these for Stage 1, Stage 2, and Stage 3.
    """
    def __init__(self, in_channels=256, num_classes=20, num_convs=4, prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Parallel Convolutional Branches
        # Note: We use GroupNorm (GN) instead of BatchNorm. BN requires large batch sizes
        # to estimate statistics accurately, which is rare in heavy object detection.
        cls_branch = []
        reg_branch = []
        for _ in range(num_convs):
            cls_branch.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_branch.append(nn.GroupNorm(32, in_channels))
            cls_branch.append(nn.ReLU(inplace=True))
            
            reg_branch.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            reg_branch.append(nn.GroupNorm(32, in_channels))
            reg_branch.append(nn.ReLU(inplace=True))
            
        self.cls_branch = nn.Sequential(*cls_branch)
        self.reg_branch = nn.Sequential(*reg_branch)
        
        # 2. Final Prediction Layers
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # 3. Initialization
        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob):
        """
        Initializes weights using Normal distributions and applies the Focal Loss
        prior bias to the classification branch to prevent early loss explosions.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        # Apply Focal Loss Prior to Classification Bias
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features):
        """
        Args:
            features (List[Tensor]): FPN features [P3, P4, P5, P6, P7]
        Returns:
            cls_scores (List[Tensor]): Classification scores per FPN level.
            bbox_preds (List[Tensor]): Bounding box (l,t,r,b) distances per FPN level.
            centerness_scores (List[Tensor]): Centerness scores per FPN level.
        """
        cls_scores = []
        bbox_preds = []
        centerness_scores = []
        
        # In FCOS, the head is shared across all FPN levels for the current stage
        for feature in features:
            cls_feat = self.cls_branch(feature)
            reg_feat = self.reg_branch(feature)
            
            # Classification
            cls_score = self.cls_logits(cls_feat)
            
            # Regression: Use F.relu to ensure predicted distances are > 0.
            # Unlike anchor-based networks that predict log-offsets, FCOS directly predicts 
            # positive pixel distances (l, t, r, b) from the location to the box edges.
            bbox_pred = torch.relu(self.bbox_pred(reg_feat))
            
            # Centerness
            centerness = self.centerness(reg_feat)
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centerness_scores.append(centerness)
            
        return cls_scores, bbox_preds, centerness_scores

# --- Usage Example ---
# fpn_features = [torch.randn(2, 256, 100, 134), torch.randn(2, 256, 50, 67)] # P3, P4 mock features
# head_stage_1 = StageSpecificFCOSHead(in_channels=256, num_classes=20)
# cls_s, bbox_s, cent_s = head_stage_1(fpn_features)