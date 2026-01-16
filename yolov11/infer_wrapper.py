import keras
from keras import ops

from .yolo import YoloV11
from .tal import make_anchors, dist2bbox, bbox_iou
from .block import DFL


class YoloV11Inference(keras.Model):
    """Inference wrapper that decodes predictions and applies Ultralytics-style NMS."""

    def __init__(
        self,
        base_model: YoloV11,
        num_classes: int,
        strides=(8, 16, 32),
        reg_max=16,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        max_nms: int = 30000,
        max_wh: float = 7680.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.dfl = DFL()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.max_nms = max_nms
        self.max_wh = max_wh

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "strides": list(self.strides),
                "reg_max": self.reg_max,
                "conf_thres": self.conf_thres,
                "iou_thres": self.iou_thres,
                "max_det": self.max_det,
                "max_nms": self.max_nms,
                "max_wh": self.max_wh,
                "base_model_config": self.base_model.get_config() if hasattr(self.base_model, "get_config") else None,
                "base_model_class": self.base_model.__class__.__name__,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        base_model_config = config.pop("base_model_config", None)
        base_model_class = config.pop("base_model_class", "YoloV11")
        strides = config.get("strides", (8, 16, 32))
        if isinstance(strides, list):
            config["strides"] = tuple(strides)
        base_model = None
        if base_model_class == "YoloV11" and base_model_config is not None:
            base_model = YoloV11.from_config(base_model_config)
        elif base_model_config is not None:
            base_model = YoloV11.from_config(base_model_config)
        else:
            base_model = YoloV11(num_classes=config.get("num_classes", 1), input_res=640)
        return cls(base_model=base_model, **config)

    @staticmethod
    def _xywh_to_xyxy(box_xywh):
        x, y, w, h = ops.split(box_xywh, 4, axis=-1)
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return ops.concatenate([x1, y1, x2, y2], axis=-1)

    def _nms_single(self, boxes_xywh, scores):
        """Greedy NMS (class-aware, Ultralytics defaults)."""
        neg_inf = ops.convert_to_tensor(-1e9, dtype=ops.dtype(scores))

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        cls_conf = ops.max(scores, axis=-1)
        cls_ids = ops.argmax(scores, axis=-1)

        cand_mask = cls_conf > self.conf_thres
        boxes = boxes_xyxy[cand_mask]
        cls_conf = cls_conf[cand_mask]
        cls_ids = cls_ids[cand_mask]

        num = ops.shape(boxes)[0]
        pad_len = ops.maximum(self.max_nms - num, 0)

        pad_boxes = ops.zeros((pad_len, 4), dtype=ops.dtype(boxes))
        pad_conf = ops.full((pad_len,), neg_inf, dtype=ops.dtype(cls_conf))
        pad_cls = ops.zeros((pad_len,), dtype=ops.dtype(cls_ids))

        boxes_padded = ops.concatenate([boxes, pad_boxes], axis=0)
        conf_padded = ops.concatenate([cls_conf, pad_conf], axis=0)
        cls_padded = ops.concatenate([cls_ids, pad_cls], axis=0)

        top_conf, top_idx = ops.top_k(conf_padded, k=self.max_nms)
        boxes_sorted = boxes_padded[top_idx]
        conf_sorted = top_conf
        cls_sorted = cls_padded[top_idx]

        valid_mask = top_idx < num
        suppressed = ops.logical_not(valid_mask)
        selected = ops.zeros_like(suppressed)

        offset = ops.cast(cls_sorted[..., None], ops.dtype(boxes_sorted)) * self.max_wh
        boxes_offset = boxes_sorted + offset

        keep_count = ops.zeros((), dtype="int32")
        cond_fn = lambda sel, sup, cnt: ops.logical_and(cnt < self.max_det, ops.any(ops.logical_not(sup)))

        def body_fn(sel, sup, cnt):
            valid_scores = ops.where(sup, neg_inf, conf_sorted)
            idx = ops.argmax(valid_scores, axis=0)
            sel = ops.logical_or(sel, ops.cast(ops.one_hot(idx, self.max_nms), "bool"))
            iou = bbox_iou(ops.expand_dims(boxes_offset[idx], 0), boxes_offset, xywh=False, CIoU=False)
            iou = ops.reshape(iou, (-1,))
            sup = ops.logical_or(sup, iou > self.iou_thres)
            cnt = cnt + 1
            return sel, sup, cnt

        selected, suppressed, keep_count = ops.while_loop(cond_fn, body_fn, (selected, suppressed, keep_count))

        selected_conf = ops.where(selected, conf_sorted, neg_inf)
        final_conf, final_idx = ops.top_k(selected_conf, k=self.max_det)
        final_valid = final_conf > self.conf_thres

        boxes_out = boxes_sorted[final_idx]
        cls_out = cls_sorted[final_idx]
        out = ops.concatenate([boxes_out, ops.expand_dims(final_conf, -1), ops.expand_dims(ops.cast(cls_out, ops.dtype(boxes_out)), -1)], axis=-1)
        out = ops.where(final_valid[:, None], out, ops.zeros_like(out))
        return out

    def call(self, images, training=None):
        preds = self.base_model(images, training=False)  # list of per-scale raw outputs
        no = self.num_classes + 4 * self.reg_max
        x_flat = [ops.reshape(xi, (ops.shape(xi)[0], -1, no)) for xi in preds]
        x_cat = ops.concatenate(x_flat, axis=1)  # (B, N, no)
        box, cls = ops.split(x_cat, [4 * self.reg_max], axis=-1)

        anchors, stride_tensor = make_anchors(preds, self.strides, grid_cell_offset=0.5)
        anchors = ops.cast(anchors, ops.dtype(cls))
        stride_tensor = ops.cast(stride_tensor, ops.dtype(cls))

        box = ops.reshape(box, (ops.shape(box)[0], ops.shape(box)[1], 4, self.reg_max))
        box = self.dfl(box)
        dbox = dist2bbox(box, ops.expand_dims(anchors, axis=0), xywh=True)
        dbox = dbox * ops.reshape(stride_tensor, (1, -1, 1))
        scores = ops.sigmoid(cls)

        outputs = []
        for b in range(ops.shape(dbox)[0]): outputs.append(self._nms_single(dbox[b], scores[b]))
        return ops.stack(outputs, axis=0)
