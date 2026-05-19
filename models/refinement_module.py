import torch
import torch.nn as nn
import torchvision.ops as ops


class LightweightFCM(nn.Module):
    """Feature Consistency Module (FCM) — box-conditioned, inverted-bottleneck DCN.

    Fixes vs. original
    ------------------
    Bug 1 (FCM is box-blind):
        The original module derived deformable-conv offsets only from the
        current feature map x.  Offsets were therefore unrelated to where the
        previous stage thought the boxes were — violating the whole point of
        cascade refinement.

        Fix: add a ``box_proj`` branch that encodes the previous stage's
        per-location (l, t, r, b) prediction map into mid_channels features.
        These are concatenated with the reduced feature and fed to the offset
        predictor, so the deformable kernel actually warps towards the refined
        box center.

        Reference: Cascade RetinaNet (Zhang et al., BMVC 2020), Sec. 4.2:
        *"the offset for each location on the feature map is predicted [from
        the regressed anchor] and a simple deformable convolution layer is
        utilised to generate the refined feature map for the following stage."*

    Bug 5 (BatchNorm in FCM):
        The rest of the model (head, FPN head branches) uses GroupNorm(32).
        BN statistics are unreliable at the small batch sizes common in
        detection (2–4 images/GPU).  Fix: replace all BatchNorm2d → GroupNorm.

    Architecture (per FPN level)
    ----------------------------
    Input  x           : (B, 256, H, W)   — stage-t FPN feature
    Input  bbox_prev   : (B,   4, H, W)   — stage-t box prediction (detached)

        reduce_conv (1×1)  : 256 → mid (64)      ┐ inverted-bottleneck
        box_proj    (1×1)  :   4 → mid (64)      │ (lecturer requirement)
        concat             : cat → 2·mid (128)   │
        offset_conv (3×3)  : 2·mid → 18          ┘ → DCN offsets
        deform_conv (3×3)  : mid → mid            offset-warped features
        expand_conv (1×1)  : mid → 256
        residual           : x + expand(·)

    Parameter delta vs. original:
        box_proj: 4 × 64 × 1 × 1 = 256 weights  (negligible)
        offset_conv input doubles (64→128): +64 × 18 × 9 = +10 368 weights.
        Total delta: ~10 624 params ≈ +0.02 % of model.  Well within budget.

    FLOP delta per level at 800-px input (P3 = 100×134):
        box_proj: 4 × 64 × 100 × 134 ≈ 3.4 M MACs.
        Across 5 levels ≈ 5 M MACs total < 0.1 % of backbone FLOPs.
    """

    def __init__(self, in_channels: int = 256, mid_channels: int = 64):
        super().__init__()

        # ── Inverted bottleneck: reduce feature channels ──────────────────
        self.reduce_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.reduce_gn   = nn.GroupNorm(32, mid_channels)   # Fix 5: was BatchNorm2d
        self.relu1       = nn.ReLU(inplace=True)

        # ── Box-conditioning branch (Fix 1) ──────────────────────────────
        # Project the 4-channel (l,t,r,b) map into mid_channels so the
        # offset predictor can learn which spatial direction the box has
        # shifted relative to the static grid point.
        self.box_proj = nn.Conv2d(4, mid_channels, kernel_size=1, bias=False)
        # Small normal init — box values can be large (pixels), but the
        # network will normalise quickly with GN downstream.
        nn.init.normal_(self.box_proj.weight, mean=0, std=0.01)

        # ── Offset predictor: takes [feat_reduced ‖ box_emb] ─────────────
        # 3×3 DCN kernel has 9 sample positions × 2 (dx, dy) = 18 offset channels.
        # Input channels: mid_channels * 2 because we concatenate feat + box_emb.
        self.offset_conv = nn.Conv2d(mid_channels * 2, 2 * 3 * 3, kernel_size=3, padding=1)
        # Zero-init ensures the DCN acts as a standard conv at training start,
        # preventing instability before the box branch has warmed up.
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias,   0)

        # ── Deformable Convolution (feature alignment) ────────────────────
        self.deform_conv = ops.DeformConv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.deform_gn   = nn.GroupNorm(32, mid_channels)   # Fix 5: was BatchNorm2d
        self.relu2       = nn.ReLU(inplace=True)

        # ── Expand channels back ──────────────────────────────────────────
        self.expand_conv = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.expand_gn   = nn.GroupNorm(32, in_channels)    # Fix 5: was BatchNorm2d

    def forward(self, x: torch.Tensor, bbox_pred_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x             : (B, in_channels, H, W)
                            FPN feature from the current cascade stage.
            bbox_pred_prev: (B, 4, H, W)   — **must be detached** by the caller.
                            The stage-t box predictions (l, t, r, b) in pixel
                            distances (i.e. after exp(scale·raw) in the head).
                            These tell the FCM in which direction each dense
                            location's box has shifted, so the deformable kernel
                            can warp the feature map toward the refined center.

        Returns:
            Tensor (B, in_channels, H, W): refined feature map for stage t+1.
        """
        identity = x  # saved for residual connection

        # ── Step 1: compress feature channels 256 → 64 ───────────────────
        feat = self.relu1(self.reduce_gn(self.reduce_conv(x)))
        # feat: (B, mid, H, W)

        # ── Step 2: encode previous box geometry → 64 channels ───────────
        # bbox_pred_prev holds (l, t, r, b) distances at each grid point.
        # Their spatial pattern encodes how far the predicted box has moved
        # from the static grid origin — exactly the information the DCN offset
        # predictor needs to "chase" the refined center.
        box_emb = self.box_proj(bbox_pred_prev)
        # box_emb: (B, mid, H, W)

        # ── Step 3: predict box-conditioned DCN offsets ───────────────────
        offset_input = torch.cat([feat, box_emb], dim=1)   # (B, 2·mid, H, W)
        offsets = self.offset_conv(offset_input)            # (B, 18, H, W)

        # ── Step 4: warp feature map with learned offsets ─────────────────
        out = self.relu2(self.deform_gn(self.deform_conv(feat, offsets)))
        # out: (B, mid, H, W)

        # ── Step 5: expand 64 → 256 ──────────────────────────────────────
        out = self.expand_gn(self.expand_conv(out))
        # out: (B, in_channels, H, W)

        # ── Step 6: residual connection ───────────────────────────────────
        return torch.relu(identity + out)
