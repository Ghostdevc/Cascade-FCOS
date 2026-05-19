import torch

class DynamicFCOSTargetGenerator:
    """
    Generates vectorized targets for Cascade FCOS. 
    Handles dynamic center sampling radiuses for different cascade stages.
    """
    def __init__(self, fpn_strides=[8, 16, 32, 64, 128]):
        self.fpn_strides = fpn_strides

    def compute_locations(self, features):
        """
        Maps FPN feature map grid points back to the original image space.
        """
        locations_per_level = []
        for level, feature in enumerate(features):
            stride = self.fpn_strides[level]
            h, w = feature.shape[2:]
            device = feature.device
            
            # Create grid coordinates
            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            
            # Map back to image space (center of the receptive field)
            shift_x = shift_x.reshape(-1) + stride // 2
            shift_y = shift_y.reshape(-1) + stride // 2
            locations_per_level.append(torch.stack((shift_x, shift_y), dim=1))
            
        return locations_per_level

    def generate_targets_for_image(self, locations_per_level, gt_boxes, gt_labels, center_radius):
        """
        Vectorized label assignment for a single image.
        Args:
            locations_per_level: List of tensors of shape (N_i, 2)
            gt_boxes: Tensor of shape (M, 4) containing ground truth boxes.
            gt_labels: Tensor of shape (M,) containing ground truth classes.
            center_radius: Float representing the dynamic R for the current cascade stage.
        """
        # Concatenate all locations across P3-P7 into a single tensor (N, 2)
        locations = torch.cat(locations_per_level, dim=0)
        num_locs = locations.shape[0]
        num_gts = gt_boxes.shape[0]

        if num_gts == 0:
            return (
                torch.zeros(num_locs, dtype=torch.int64, device=locations.device),
                torch.zeros((num_locs, 4), dtype=torch.float32, device=locations.device)
            )

        # 1. Compute Distances (l, t, r, b) for ALL locations to ALL ground truth boxes
        # Expand dims to compute pairwise: locations (N, 1, 2) and gt_boxes (1, M, 4)
        xs, ys = locations[:, 0], locations[:, 1]
        
        # Calculate l, t, r, b matrices of shape (N, M)
        l = xs[:, None] - gt_boxes[:, 0][None, :]
        t = ys[:, None] - gt_boxes[:, 1][None, :]
        r = gt_boxes[:, 2][None, :] - xs[:, None]
        b = gt_boxes[:, 3][None, :] - ys[:, None]
        
        reg_targets = torch.stack([l, t, r, b], dim=-1) # Shape: (N, M, 4)
        
        # Mask 1: Is the location inside the ground truth box?
        is_in_boxes = reg_targets.min(dim=-1)[0] > 0 # Shape: (N, M)

        # 2. Dynamic Center Sampling Mask
        # Compute the center of each ground truth box
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        
        # We need the specific stride for each location to compute the valid center sub-box
        strides = torch.cat([
            torch.full((loc.shape[0],), stride, device=loc.device) 
            for loc, stride in zip(locations_per_level, self.fpn_strides)
        ])
        
        # Calculate the dynamic center bounding box using the current stage's radius
        center_bboxes_l = gt_cx[None, :] - center_radius * strides[:, None]
        center_bboxes_t = gt_cy[None, :] - center_radius * strides[:, None]
        center_bboxes_r = gt_cx[None, :] + center_radius * strides[:, None]
        center_bboxes_b = gt_cy[None, :] + center_radius * strides[:, None]
        
        # Clip the center box to the boundaries of the actual ground truth box
        center_bboxes_l = torch.max(center_bboxes_l, gt_boxes[:, 0][None, :])
        center_bboxes_t = torch.max(center_bboxes_t, gt_boxes[:, 1][None, :])
        center_bboxes_r = torch.min(center_bboxes_r, gt_boxes[:, 2][None, :])
        center_bboxes_b = torch.min(center_bboxes_b, gt_boxes[:, 3][None, :])
        
        c_l = xs[:, None] - center_bboxes_l
        c_t = ys[:, None] - center_bboxes_t
        c_r = center_bboxes_r - xs[:, None]
        c_b = center_bboxes_b - ys[:, None]
        
        center_reg_targets = torch.stack([c_l, c_t, c_r, c_b], dim=-1)
        
        # Mask 2: Is the location inside the center sampling sub-box?
        is_in_centers = center_reg_targets.min(dim=-1)[0] > 0 # Shape: (N, M)

        # 3. Handle Ambiguity (Locations falling into multiple GT boxes)
        # If a location falls into multiple ground-truth boxes, we choose the object with minimal area [cite: 65]
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        
        # Combine masks
        valid_mask = is_in_boxes & is_in_centers # Shape: (N, M)
        
        # To find the valid box with the smallest area per location, we use a trick:
        # Give invalid boxes an infinite area so they are never chosen
        areas_matrix = gt_areas[None, :].repeat(num_locs, 1)
        areas_matrix[~valid_mask] = float('inf')
        
        min_areas, min_area_indices = areas_matrix.min(dim=1) # Shape: (N,)
        
        # Identify locations that do not match any ground truth box
        background_mask = min_areas == float('inf')
        
        # 4. Finalizing Targets
        matched_gt_labels = gt_labels[min_area_indices]
        matched_gt_labels[background_mask] = 0 # 0 denotes background
        
        # Gather the valid regression targets based on the chosen GT box index
        batch_indices = torch.arange(num_locs, device=locations.device)
        matched_reg_targets = reg_targets[batch_indices, min_area_indices]
        
        return matched_gt_labels, matched_reg_targets