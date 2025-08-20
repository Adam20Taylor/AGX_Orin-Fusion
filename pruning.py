import numpy as np

def bev_boxes_np_to_2d(boxes_np):
    """
    Converts bounding boxes from (x, y, z, dx, dy, dz, heading) to (x1, y1, x2, y2)
    Assumes axis-aligned (ignores heading).
    """
    x, y, dx, dy = boxes_np[0], boxes_np[1], boxes_np[3], boxes_np[4]
    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou_2d(boxes1_np, boxes2_np):
    """
    boxes1_np, boxes2_np: (N, 7) and (M, 7) numpy arrays (x, y, z, dx, dy, dz, heading)
    Returns: IoU matrix (N, M)
    """
    boxes1 = bev_boxes_np_to_2d(boxes1_np)
    boxes2 = bev_boxes_np_to_2d(boxes2_np)
    # Intersection coords
    inter_x1 = max(boxes1[0], boxes2[0])
    inter_y1 = max(boxes1[1], boxes2[1])
    inter_x2 = min(boxes1[2], boxes2[2])
    inter_y2 = min(boxes1[3], boxes2[3])

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    area1 = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    area2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    union_area = area1 + area2 - inter_area

    return inter_area / union_area


def prune(bboxes, scores, iou_threshold=0.25):
    if len(bboxes) == 0:
        return np.array([]), np.array([])

    # Sort by confidence score descending
    order = np.argsort(-scores)
    bboxes = bboxes[order]
    scores = scores[order]

    keep_boxes = []
    keep_scores = []

    while len(bboxes) > 0:
        # Take the box with the highest score
        curr_box = bboxes[0]
        curr_score = scores[0]
        keep_boxes.append(curr_box)
        keep_scores.append(curr_score)

        # Compute IoU of the rest with the current box
        rest_boxes = bboxes[1:]
        if len(rest_boxes) == 0:
            break

        ious = np.array([iou_2d(curr_box, box) for box in rest_boxes])

        # Keep boxes with IoU less than threshold
        keep_indices = np.where(ious < iou_threshold)[0]
        
        # Update bboxes and scores to only include boxes below threshold
        bboxes = rest_boxes[keep_indices]
        scores = scores[1:][keep_indices]

    return np.array(keep_boxes), np.array(keep_scores)
