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


class TaskAlignedAssigner:
    """Task-aligned assigner rewritten with keras.ops (channels-last friendly)."""

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        bs = pd_scores.shape[0]
        n_max_boxes = gt_bboxes.shape[1]

        if n_max_boxes == 0:
            return (
                ops.full_like(pd_scores[..., 0], self.num_classes),
                ops.zeros_like(pd_bboxes),
                ops.zeros_like(pd_scores),
                ops.zeros_like(pd_scores[..., 0]),
                ops.zeros_like(pd_scores[..., 0]),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask, bs, n_max_boxes)

        align_metric = align_metric * mask_pos
        pos_align_metrics = ops.max(align_metric, axis=-1, keepdims=True)
        pos_overlaps = ops.max(overlaps * mask_pos, axis=-1, keepdims=True)
        norm_align_metric = ops.max(align_metric * pos_overlaps / (pos_align_metrics + self.eps), axis=-2, keepdims=True)
        # reshape to (bs, na, 1) to align with target_scores (bs, na, C)
        norm_align_metric = ops.transpose(norm_align_metric, (0, 2, 1))
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask > 0, target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=ops.broadcast_to(mask_gt, align_metric.shape[:-1] + (self.topk,)))
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        bs = pd_scores.shape[0]
        na = pd_bboxes.shape[-2]
        n_max = gt_labels.shape[1]
        mask_gt = mask_gt.astype("bool")

        overlaps = ops.zeros((bs, n_max, na), dtype=pd_bboxes.dtype)
        bbox_scores = ops.zeros((bs, n_max, na), dtype=pd_scores.dtype)

        cls_idx = ops.squeeze(gt_labels, axis=-1)
        cls_idx = ops.where(cls_idx < 0, 0, cls_idx)
        cls_one_hot = ops.one_hot(cls_idx, num_classes=pd_scores.shape[-1], dtype=pd_scores.dtype)  # (b, n_max, C)
        cls_one_hot = ops.expand_dims(cls_one_hot, axis=2)  # (b, n_max,1,C)
        cls_one_hot = ops.broadcast_to(cls_one_hot, (bs, n_max, na, pd_scores.shape[-1]))
        bbox_scores = ops.sum(cls_one_hot * ops.expand_dims(pd_scores, 1), axis=-1)
        bbox_scores = ops.where(mask_gt, bbox_scores, ops.zeros_like(bbox_scores))

        pd_boxes = ops.expand_dims(pd_bboxes, axis=1)
        pd_boxes = ops.broadcast_to(pd_boxes, (bs, n_max, na, 4))
        gt_boxes = ops.expand_dims(gt_bboxes, axis=2)
        gt_boxes = ops.broadcast_to(gt_boxes, (bs, n_max, na, 4))
        pd_boxes = ops.where(mask_gt[..., None], pd_boxes, ops.zeros_like(pd_boxes))
        gt_boxes = ops.where(mask_gt[..., None], gt_boxes, ops.zeros_like(gt_boxes))

        overlaps = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True)
        overlaps = ops.where(mask_gt, overlaps, 0)

        align_metric = ops.power(bbox_scores, self.alpha) * ops.power(overlaps, self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        topk_metrics, topk_idxs = ops.top_k(metrics, k=self.topk, sorted=True)
        if topk_mask is None:
            topk_mask = ops.broadcast_to(ops.max(topk_metrics, axis=-1, keepdims=True) > self.eps, topk_idxs.shape)
        topk_idxs = ops.where(topk_mask, topk_idxs, ops.zeros_like(topk_idxs))

        num_anchors = metrics.shape[-1]
        one_hot = ops.one_hot(topk_idxs, num_classes=num_anchors)
        count_tensor = ops.sum(one_hot, axis=-2)
        count_tensor = ops.where(count_tensor > 1, ops.zeros_like(count_tensor), count_tensor)
        return count_tensor.astype(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, bs, n_max_boxes):
        batch_ind = ops.arange(bs, dtype="int32")[:, None]
        flat_idx = target_gt_idx + batch_ind * n_max_boxes
        flat_labels = ops.reshape(gt_labels, (-1,))
        target_labels = ops.take(flat_labels, flat_idx)
        target_labels = ops.maximum(target_labels, 0)

        flat_boxes = ops.reshape(gt_bboxes, (-1, 4))
        target_bboxes = ops.take(flat_boxes, flat_idx, axis=0)

        target_scores = ops.one_hot(target_labels, num_classes=self.num_classes, dtype=gt_bboxes.dtype)
        target_scores = ops.where(fg_mask[..., None] > 0, target_scores, ops.zeros_like(target_scores))
        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        n_anchors = xy_centers.shape[0]
        lt, rb = ops.split(gt_bboxes, 2, axis=-1)
        bbox_deltas = ops.concatenate(
            [ops.expand_dims(xy_centers, 0) - lt[..., None, :], rb[..., None, :] - ops.expand_dims(xy_centers, 0)],
            axis=-1,
        )
        bbox_deltas = ops.reshape(bbox_deltas, (*gt_bboxes.shape[:2], n_anchors, 4))
        return ops.min(bbox_deltas, axis=-1) > eps

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        fg_mask = ops.sum(mask_pos, axis=-2)
        mask_multi_gts = fg_mask > 1
        max_overlaps_idx = ops.argmax(overlaps, axis=1)
        is_max_overlaps = ops.transpose(ops.one_hot(max_overlaps_idx, num_classes=n_max_boxes), (0, 2, 1))
        mask_pos = ops.where(mask_multi_gts[:, None, :], is_max_overlaps, mask_pos).astype("float32")
        fg_mask = ops.sum(mask_pos, axis=-2)
        target_gt_idx = ops.argmax(mask_pos, axis=-2)
        return target_gt_idx, fg_mask, mask_pos


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
