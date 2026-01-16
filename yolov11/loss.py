import keras
from keras import ops, layers

from .tal import TaskAlignedAssigner, dist2bbox, bbox2dist, bbox_iou


def dfl_loss(pred_dist, target, reg_max):
    """Distribution Focal Loss for one side. pred_dist: (..., reg_max) logits; target: (...)."""
    target = ops.clip(target, 0.0, reg_max - 1 - 0.01)
    tl = ops.floor(target)
    tr = tl + 1.0
    wl = tr - target
    wr = 1.0 - wl
    log_probs = ops.log(ops.softmax(pred_dist, axis=-1) + 1e-9)
    tl = ops.cast(tl, "int32")
    tr = ops.cast(tr, "int32")
    log_tl = ops.take_along_axis(log_probs, tl[..., None], axis=-1)[..., 0]
    log_tr = ops.take_along_axis(log_probs, tr[..., None], axis=-1)[..., 0]
    loss = -(log_tl * wl + log_tr * wr)
    return ops.mean(loss, axis=-1, keepdims=True)


def _decode_bboxes(pred_dist, anchors, reg_max):
    """Project distribution to ltrb and convert to xyxy using anchors in grid units."""
    b = ops.shape(pred_dist)[0]
    n = ops.shape(pred_dist)[1]
    dist = ops.reshape(pred_dist, (b, n, 4, reg_max))
    proj = ops.cast(ops.arange(reg_max), ops.dtype(dist))
    dist = ops.sum(ops.softmax(dist, axis=-1) * proj, axis=-1)  # (B,N,4)
    return dist2bbox(dist, anchors, xywh=False)


def _prepare_predictions(y_pred, num_classes, reg_max):
    """Return pred_dist, pred_scores, anchors (N,2), stride_tensor (N,1) from packed output tensor."""
    no = num_classes + 4 * reg_max
    x_cat = y_pred[..., :no]
    anchors = ops.stop_gradient(y_pred[0, :, no : no + 2])
    stride_tensor = ops.stop_gradient(y_pred[0, :, no + 2 : no + 3])
    pred_dist, pred_scores = ops.split(x_cat, [4 * reg_max], axis=-1)
    return pred_dist, pred_scores, anchors, stride_tensor


def build_yolo_detection_loss(num_classes, reg_max=16, strides=(8, 16, 32), topk=10, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5):
    """Factory returning YOLOv11 detection loss callable (DFL + CIoU + BCE) using TaskAlignedAssigner."""
    assigner = TaskAlignedAssigner(topk=topk, num_classes=num_classes, alpha=0.5, beta=6.0)

    def loss_fn(y_true, y_pred):
        gt_labels = ops.expand_dims(ops.cast(y_true[..., 0], "int32"), axis=-1)
        gt_bboxes = y_true[..., 1:5]
        pred_dist, pred_scores, anchors, stride_tensor = _prepare_predictions(y_pred, num_classes=num_classes, reg_max=reg_max)

        anchors = ops.stop_gradient(ops.cast(anchors, ops.dtype(pred_scores)))
        stride_tensor = ops.stop_gradient(ops.cast(stride_tensor, ops.dtype(pred_scores)))

        pred_bboxes = _decode_bboxes(pred_dist, ops.expand_dims(anchors, axis=0), reg_max)
        stride_b = ops.reshape(stride_tensor, (1, -1, 1))

        mask_gt = ops.sum(gt_bboxes, axis=-1, keepdims=True) > 0
        pd_scores_sig = ops.stop_gradient(ops.sigmoid(pred_scores))
        ta_labels, ta_bboxes, ta_scores, fg_mask, _ = assigner(
            pd_scores_sig,
            pred_bboxes * stride_b,
            anchors * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = ops.maximum(ops.sum(ta_scores), 1.0)

        # Match Ultralytics BCE reduction: include negatives, normalize by positive score sum
        cls_bce = ops.binary_crossentropy(ta_scores, pred_scores, from_logits=True)
        cls_loss = ops.sum(cls_bce) / target_scores_sum

        fg_mask_flat = ops.reshape(fg_mask, (-1,))
        pred_b_flat = ops.reshape(pred_bboxes, (-1, 4))
        tgt_b_flat = ops.reshape(ta_bboxes / stride_b, (-1, 4))
        weights_flat = ops.reshape(ops.sum(ta_scores, axis=-1, keepdims=True), (-1, 1))

        mask_w = ops.cast(fg_mask_flat[:, None], ops.dtype(weights_flat))
        weights_fg = weights_flat * mask_w

        iou = bbox_iou(tgt_b_flat, pred_b_flat, xywh=False, CIoU=True)
        box_loss = ops.sum((1.0 - iou) * weights_fg[..., 0]) / target_scores_sum

        pred_dist_flat = ops.reshape(pred_dist, (-1, 4, reg_max))
        pred_dist_vec = ops.reshape(pred_dist_flat, (-1, reg_max))
        target_ltrb = bbox2dist(anchors, ta_bboxes / stride_tensor, reg_max - 1)
        target_ltrb_flat = ops.reshape(target_ltrb, (-1, 4))
        target_ltrb_vec = ops.reshape(target_ltrb_flat, (-1,))

        weights_dfl = ops.reshape(ops.tile(weights_fg, (1, 4)), (-1, 1))
        dfl = dfl_loss(pred_dist_vec, target_ltrb_vec, reg_max)
        dfl_loss_val = ops.sum(dfl * weights_dfl) / target_scores_sum

        return box_gain * box_loss + cls_gain * cls_loss + dfl_gain * dfl_loss_val

    return loss_fn


class YoloDetectionLoss(layers.Layer):
    """Layer wrapper kept for compatibility; delegates to build_yolo_detection_loss output."""

    def __init__(self, num_classes, reg_max=16, strides=(8, 16, 32), topk=10, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5, **kwargs):
        super().__init__(**kwargs)
        self._loss_fn = build_yolo_detection_loss(
            num_classes=num_classes,
            reg_max=reg_max,
            strides=strides,
            topk=topk,
            box_gain=box_gain,
            cls_gain=cls_gain,
            dfl_gain=dfl_gain,
        )

    def call(self, inputs):
        preds, gt_labels, gt_bboxes = inputs
        return self._loss_fn((gt_labels, gt_bboxes), preds)
