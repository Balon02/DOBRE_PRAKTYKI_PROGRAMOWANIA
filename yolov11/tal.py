import keras
from keras import ops, layers


def bbox_iou(box1, box2, xywh=False, CIoU=False, eps=1e-7):
    """Compute IoU or CIoU between box1 and box2 (both shape (..., 4) xyxy unless xywh=True)."""
    if xywh:
        # convert xywh to xyxy
        x, y, w, h = ops.split(box1, 4, axis=-1)
        box1 = ops.concatenate([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], axis=-1)
        x, y, w, h = ops.split(box2, 4, axis=-1)
        box2 = ops.concatenate([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], axis=-1)

    # Intersection
    inter_lt = ops.maximum(box1[..., :2], box2[..., :2])
    inter_rb = ops.minimum(box1[..., 2:], box2[..., 2:])
    inter_wh = ops.maximum(inter_rb - inter_lt, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    if not CIoU:
        return iou

    # CIoU components
    pi = ops.convert_to_tensor(3.141592653589793, dtype=iou.dtype)
    cw = ops.maximum(box1[..., 2], box2[..., 2]) - ops.minimum(box1[..., 0], box2[..., 0])
    ch = ops.maximum(box1[..., 3], box2[..., 3]) - ops.minimum(box1[..., 1], box2[..., 1])
    c2 = cw * cw + ch * ch + eps
    rho2 = (
        (box2[..., 0] + box2[..., 2] - box1[..., 0] - box1[..., 2]) ** 2
        + (box2[..., 1] + box2[..., 3] - box1[..., 1] - box1[..., 3]) ** 2
    ) / 4
    ar1 = (box2[..., 2] - box2[..., 0]) / (box2[..., 3] - box2[..., 1] + eps)
    ar2 = (box1[..., 2] - box1[..., 0]) / (box1[..., 3] - box1[..., 1] + eps)
    v = (4 / (pi ** 2)) * ops.square(ops.arctan(ar1) - ops.arctan(ar2))
    alpha = ops.stop_gradient(v / (1 - iou + v + eps))
    return iou - (rho2 / c2 + v * alpha)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype = ops.dtype(feats[0])
    for f, stride in zip(feats, strides):
        h = ops.shape(f)[1]
        w = ops.shape(f)[2]
        sx = ops.arange(w, dtype=dtype) + grid_cell_offset
        sy = ops.arange(h, dtype=dtype) + grid_cell_offset
        sy, sx = ops.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(ops.reshape(ops.stack([sx, sy], axis=-1), (-1, 2)))
        stride_tensor.append(ops.full((h * w, 1), stride, dtype=dtype))
    return ops.concatenate(anchor_points, axis=0), ops.concatenate(stride_tensor, axis=0)


def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
    lt, rb = ops.split(distance, 2, axis=axis)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) * 0.5
        wh = x2y2 - x1y1
        return ops.concatenate([c_xy, wh], axis=axis)
    return ops.concatenate([x1y1, x2y2], axis=axis)


def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = ops.split(bbox, 2, axis=-1)
    return ops.clip(ops.concatenate([anchor_points - x1y1, x2y2 - anchor_points], axis=-1), 0, reg_max - 0.01)
