import keras
from keras import ops

from .yolo import YoloV11
from .tal import make_anchors


class YoloV11Training(keras.models.Model):
    """Training wrapper that packs raw heads with anchor metadata for loss_fn consumption."""

    def __init__(self, base_model: YoloV11, num_classes: int, strides=(8, 16, 32), reg_max: int | None = None, input_res: int = 640, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max if reg_max is not None else getattr(base_model.detect, "reg_max", 16)
        self._input_res = input_res

    def call(self, inputs, training=None):
        if training is None: training = False
        images = inputs
        preds = self.base_model(images, training=training)  # list of per-scale raw outputs

        no = self.num_classes + 4 * self.reg_max
        x_flat = [ops.reshape(xi, (ops.shape(xi)[0], -1, no)) for xi in preds]
        x_cat = ops.concatenate(x_flat, axis=1)  # (B, N, no)

        anchors, stride_tensor = make_anchors(preds, self.strides, grid_cell_offset=0.5)
        anchors = ops.cast(anchors, x_cat.dtype)
        stride_tensor = ops.cast(stride_tensor, x_cat.dtype)

        batch = ops.shape(x_cat)[0]
        anchors_b = ops.broadcast_to(ops.expand_dims(anchors, 0), (batch, ops.shape(anchors)[0], 2))
        stride_b = ops.broadcast_to(ops.expand_dims(stride_tensor, 0), (batch, ops.shape(stride_tensor)[0], 1))
        anchors_b = ops.stop_gradient(anchors_b)
        stride_b = ops.stop_gradient(stride_b)

        return ops.concatenate([x_cat, anchors_b, stride_b], axis=-1)
