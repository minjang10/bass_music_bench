from scipy.optimize import linear_sum_assignment
import numpy as np

def temporal_iou(seg1, seg2):
    """
    Calculate IoU for two temporal segments
    seg1, seg2: (start_time, end_time) tuples
    """
    start1, end1 = seg1
    start2, end2 = seg2
    
    # Intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0


def calculate_iou_hungarian(pred_intervals, pred_labels, gt_intervals, gt_labels, 
                          require_label_match=True):
    """
    Calculate IoU using Hungarian algorithm for optimal assignment
    
    Args:
        pred_intervals: np.array of shape (N, 2) - prediction intervals
        pred_labels: np.array of shape (N,) - prediction labels  
        gt_intervals: np.array of shape (M, 2) - ground truth intervals
        gt_labels: np.array of shape (M,) - ground truth labels
        require_label_match: bool - whether segments must have same label to match
    
    Returns:
        mean_iou: float - mean IoU of optimal assignment
        assignments: list of tuples - (pred_idx, gt_idx, iou) for each match
    """
    n_pred = len(pred_intervals)
    n_gt = len(gt_intervals)
    
    # Handle edge cases
    if n_pred == 0 and n_gt == 0:
        return 1.0, []
    if n_pred == 0 or n_gt == 0:
        return 0.0, []
    
    # Create cost matrix (1 - IoU for minimization)
    cost_matrix = np.ones((n_pred, n_gt))  # Initialize with max cost
    
    for i in range(n_pred):
        for j in range(n_gt):
            # Check if labels match (if required)
            if require_label_match and pred_labels[i] != gt_labels[j]:
                cost_matrix[i, j] = 1.0  # Max cost for mismatched labels
            else:
                iou = temporal_iou(pred_intervals[i], gt_intervals[j])
                cost_matrix[i, j] = 1 - iou  # Convert IoU to cost
    
    # Find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate results
    total_iou = 0
    assignments = []
    
    for i, j in zip(pred_indices, gt_indices):
        cost = cost_matrix[i, j]
        if cost < 1.0:  # Valid match (not max cost)
            iou = 1 - cost
            total_iou += iou
            assignments.append((i, j, iou))
        else:
            assignments.append((i, j, 0.0))  # No valid match
    
    mean_iou = total_iou / max(n_pred, n_gt)
    
    return mean_iou, assignments