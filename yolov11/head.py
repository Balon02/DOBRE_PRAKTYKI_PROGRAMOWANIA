import math
import keras
from keras import ops, layers

from .block import YoloConv2D, YoloDWConv2D, C3k2, SPPF, C2PSA, DFL

from typing import Iterable

class YoloDetectionHead(layers.Layer):
    def __init__(self, num_classes:int=1, strides:Iterable=ops.array([8,16,32]), input_res:int=1024, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.strides = strides
        self.input_res = input_res
        self.reg_max = 16

        self.bias_iterator = 0 # maybe a little overcomplicated solution, but should work as expected

    def make_bias_init_fn(self):
        base = ops.array(math.log(5 / self.num_classes / (self.input_res / self.strides[self.bias_iterator]) ** 2))
        self.bias_iterator += 1
        def init_fn(shape, dtype=None):
            b = ops.cast(base, dtype or ops.float32)
            return ops.broadcast_to(b, shape)
        return init_fn
        
    def build(self, shape):
        assert len(shape) == len(self.strides)
        det_channels = [n[-1] for n in shape]
        self.num_detection_layers = len(det_channels)
        c1_filters, c2_filters = max((16, det_channels[0] // 4, self.reg_max * 4)), max(det_channels[0], min(self.num_classes, 100))
        self.num_outs_per_anchor = self.reg_max * 4

        self.conv1_paths = [
            keras.Sequential([
                YoloConv2D(filters=c1_filters, kernel_size=3), 
                YoloConv2D(filters=c1_filters, kernel_size=3), 
                layers.Conv2D(filters=4*self.reg_max, kernel_size=1, bias_initializer='ones')
        ])for x in det_channels]

        self.conv2_paths = [
            keras.Sequential([
                keras.Sequential([YoloDWConv2D(filters=x, kernel_size=3), YoloConv2D(filters=c2_filters, kernel_size=1)]),
                keras.Sequential([YoloDWConv2D(filters=c2_filters, kernel_size=3), YoloConv2D(filters=c2_filters, kernel_size=1)]),
                layers.Conv2D(filters=self.num_classes, kernel_size=1, bias_initializer=self.make_bias_init_fn())
        ])for x in det_channels]

        self.dfl = DFL()

    def call(self, inputs, training=None):
        train_outs = []
        for i in range(self.num_detection_layers):
            c1_out = self.conv1_paths[i](inputs[i], training=training)
            c2_out = self.conv2_paths[i](inputs[i], training=training)
            train_outs.append(ops.concatenate([c1_out, c2_out], axis=-1))
        return train_outs

    # CODEX-WRITTEN
    @staticmethod
    def _make_anchors(feats, strides, offset=0.5):
        dtype = ops.dtype(feats[0])
        anchors, stride_out = [], []
        for f, s in zip(feats, strides):  # f: (B,H,W,C)
            h = ops.shape(f)[1]; w = ops.shape(f)[2]
            sy = ops.arange(h, dtype=dtype) + offset
            sx = ops.arange(w, dtype=dtype) + offset
            gy, gx = ops.meshgrid(sy, sx, indexing="ij")
            grid = ops.stack([gx, gy], axis=-1)                       # (H,W,2) in grid units
            anchors.append(ops.reshape(grid, (-1, 2)))                # (H*W,2)
            stride_out.append(ops.full((h * w, 1), s, dtype=dtype))
        return ops.concatenate(anchors, axis=0), ops.concatenate(stride_out, axis=0)


    def get_config(self):
        config = super().get_config()
        config.update({"input_res": self.input_res, "strides": self.strides, "num_classes": self.num_classes})
        return config

