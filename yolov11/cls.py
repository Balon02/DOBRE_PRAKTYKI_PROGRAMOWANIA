import keras
from keras import ops, layers

from .block import YoloConv2D, C3k2, SPPF, C2PSA
from .head import YoloClassificationHead


class YoloV11Cls(layers.Layer):
    """YOLOv11 classification backbone/head (detector backbone reused 1:1)."""

    def __init__(
        self,
        depth: float = 0.5,
        width: float = 0.25,
        max_channels: int = 1024,
        num_classes: int = 1000,
        input_res: int = 640,
        add_downsample: bool = False,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.width = width
        self.max_channels = max_channels
        self.input_res = input_res
        self.num_classes = num_classes
        self.add_downsample = add_downsample

        # BACKBONE STACK (identical to detector)
        self.backbone_stage_1 = keras.Sequential(
            [
                YoloConv2D(filters=self._scale_width(32), kernel_size=3, strides=2)
                if add_downsample
                else layers.Identity(),
                YoloConv2D(filters=self._scale_width(64), kernel_size=3, strides=2),
                YoloConv2D(filters=self._scale_width(128), kernel_size=3, strides=2),
                *[
                    C3k2(dim=self._scale_width(256), use_c3k=False, expansion_ratio=0.25)
                    for _ in range(self._scale_depth(2))
                ],
                YoloConv2D(filters=self._scale_width(256), kernel_size=3, strides=2),
                *[
                    C3k2(dim=self._scale_width(512), use_c3k=False, expansion_ratio=0.25)
                    for _ in range(self._scale_depth(2))
                ],
            ]
        )

        self.backbone_stage_2 = keras.Sequential(
            [
                YoloConv2D(filters=self._scale_width(512), kernel_size=3, strides=2),
                *[
                    C3k2(dim=self._scale_width(512), use_c3k=True)
                    for _ in range(self._scale_depth(2))
                ],
            ]
        )

        self.backbone_stage_3 = keras.Sequential(
            [
                YoloConv2D(filters=self._scale_width(1024), kernel_size=3, strides=2),
                *[
                    C3k2(dim=self._scale_width(1024), use_c3k=True)
                    for _ in range(self._scale_depth(2))
                ],
                SPPF(dim=self._scale_width(1024), kernel_size=5),
                *[C2PSA(dim=self._scale_width(1024)) for _ in range(self._scale_depth(2))],
            ]
        )

        # CLASSIFICATION HEAD
        self.cls_head = YoloClassificationHead(
            num_classes=self.num_classes,
            dropout_rate=dropout_rate,
        )

    def _scale_width(self, dim):
        return min(int(dim * self.width), self.max_channels)

    def _scale_depth(self, repeats):
        return max(int(repeats * self.depth), 1)

    def call(self, x, training=None):
        if training is None:
            training = False
        # BACKBONE STAGES (identical to detector)
        backbone_stage_1_out = self.backbone_stage_1(x, training=training)
        backbone_stage_2_out = self.backbone_stage_2(backbone_stage_1_out, training=training)
        backbone_stage_3_out = self.backbone_stage_3(backbone_stage_2_out, training=training)
        # CLASSIFICATION HEAD consumes the deepest feature
        return self.cls_head(backbone_stage_3_out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "width": self.width,
                "max_channels": self.max_channels,
                "input_res": self.input_res,
                "num_classes": self.num_classes,
                "add_downsample": self.add_downsample,
            }
        )
        return config

