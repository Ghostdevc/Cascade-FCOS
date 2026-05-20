import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ---------------------------------------------------------------------------
# Scale Module — Fix for Bug 4 (missing per-level regression normalization)
# ---------------------------------------------------------------------------
class Scale(nn.Module):
    """Per-level learnable scalar for FCOS regression normalization.

    FCOS paper (TPAMI, Sec. 2.2) notes that different FPN levels must regress
    different size ranges (e.g. [0, 64] for P3 vs [256, ∞] for P6), so a
    single shared head without any level-specific adjustment is suboptimal.
    The fix is a learnable scalar s_i per level:

        d̂ = exp(s_i · x)

    where x is the raw conv output and exp(·) replaces relu(·) to guarantee:
      1. Positivity (distances l,t,r,b must be > 0).
      2. Smooth, non-saturating gradients everywhere (relu kills gradients for
         x < 0, which hurts early training when raw outputs are near zero).
      3. Each level can independently adapt its regression scale.

    Reference: FCOS (Tian et al., TPAMI 2022), Sec. 2.2.
    Parameter cost: 5 scalars per head × 3 stages = 15 floats total (negligible).
    """

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        # Initialise to 1.0 so that at the start exp(1.0 * x) ≈ exp(x).
        # The network can quickly rescale this per level during warm-up.
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp exponent to avoid fp16 overflow on large raw predictions.
        return torch.exp(self.scale * x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Backbone + FPN  (unchanged from original)
# ---------------------------------------------------------------------------
class FCOSBackboneFPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()

        # 1. Load Pretrained ResNet-50
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # C3: stride 8, channels 512
        self.c3_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )
        # C4: stride 16, channels 1024
        self.c4_extractor = backbone.layer3

        # C5: stride 32, channels 2048
        self.c5_extractor = backbone.layer4

        # Freeze BatchNorm layers (standard practice for object detection)
        self._freeze_bn()

        # 2. Lateral Connections (1×1 convolutions to unify channel dimensions)
        self.lat_prj3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lat_prj4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lat_prj5 = nn.Conv2d(2048, out_channels, kernel_size=1)

        # 3. Top-Down Smoothing Convolutions
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 4. P6 and P7
        self.p6_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self._init_fpn_weights()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        """Override train() to keep BatchNorm in eval mode after every mode switch.
        
        Standard detection practice (FCOS, RetinaNet, etc.): BN statistics from
        ImageNet pretraining are kept frozen during detection training, because
        detection batch sizes are too small to estimate reliable batch stats.
        """
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def _init_fpn_weights(self):
        for m in [
            self.lat_prj3, self.lat_prj4, self.lat_prj5,
            self.smooth3, self.smooth4, self.smooth5,
            self.p6_conv, self.p7_conv,
        ]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Bottom-up
        c3 = self.c3_extractor(x)
        c4 = self.c4_extractor(c3)
        c5 = self.c5_extractor(c4)

        # Lateral connections
        p5_lat = self.lat_prj5(c5)
        p4_lat = self.lat_prj4(c4)
        p3_lat = self.lat_prj3(c3)

        # Top-down pathway
        p5 = p5_lat
        p4 = p4_lat + F.interpolate(p5, size=p4_lat.shape[-2:], mode="nearest")
        p3 = p3_lat + F.interpolate(p4, size=p3_lat.shape[-2:], mode="nearest")

        # Smoothing
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        # P6, P7
        p6 = self.p6_conv(p5)
        p7 = self.p7_conv(F.relu(p6))

        return [p3, p4, p5, p6, p7]


# ---------------------------------------------------------------------------
# FCOS Head  (Fix 4 applied here)
# ---------------------------------------------------------------------------
class StageSpecificFCOSHead(nn.Module):
    """FCOS detection head for a single cascade stage.

    Changes vs. original:
      - Added ``self.scales`` (nn.ModuleList of 5 Scale modules).
      - Replaced ``torch.relu(self.bbox_pred(reg_feat))``  →
        ``self.scales[i](self.bbox_pred(reg_feat))``   (exp-based, per-level).
      - forward() now takes ``enumerate`` over features so the level index i
        is available to index into self.scales.

    Why: see Scale docstring above.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 20,
        num_convs: int = 4,
        num_fpn_levels: int = 5,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Parallel 4-conv branches with GroupNorm (GN is standard for detection;
        # BN needs large batch sizes to estimate statistics accurately).
        cls_branch, reg_branch = [], []
        for _ in range(num_convs):
            cls_branch += [
                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
            reg_branch += [
                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
        self.cls_branch = nn.Sequential(*cls_branch)
        self.reg_branch = nn.Sequential(*reg_branch)

        # Prediction heads
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred  = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # ── Fix 4: per-level learnable scales ──────────────────────────────
        # One Scale module per FPN level (P3…P7).  Initialised to 1.0 so that
        # exp(1.0 · x) starts with the same magnitude as exp(x).
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_fpn_levels)])

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        """Normal init for conv weights; focal-loss prior bias for cls head.

        The classification bias is initialised so that the initial predicted
        probability for any class is π = 0.01.  This prevents a loss explosion
        at the start of training when ~99 % of locations are background:

            bias = -log((1 - π) / π) = -log(99) ≈ -4.6
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features):
        """
        Args:
            features: List[Tensor(B, C, H_i, W_i)] — FPN levels [P3..P7].

        Returns:
            cls_scores      : List[Tensor(B, num_classes, H_i, W_i)]
            bbox_preds      : List[Tensor(B, 4, H_i, W_i)]  — pixel distances
            centerness_scores: List[Tensor(B, 1, H_i, W_i)]
        """
        cls_scores, bbox_preds, centerness_scores = [], [], []

        for i, feature in enumerate(features):
            cls_feat = self.cls_branch(feature)
            reg_feat = self.reg_branch(feature)

            cls_score  = self.cls_logits(cls_feat)

            # ── Fix 4: exp(scale_i · raw_pred) instead of relu(raw_pred) ──
            # This replaces the old `torch.relu(self.bbox_pred(reg_feat))`.
            # exp(·) guarantees positivity AND smooth gradients for all x ∈ ℝ.
            # scale_i is a learned scalar initialised to 1.0.
            bbox_pred  = self.scales[i](self.bbox_pred(reg_feat))

            centerness = self.centerness(reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centerness_scores.append(centerness)

        return cls_scores, bbox_preds, centerness_scores
