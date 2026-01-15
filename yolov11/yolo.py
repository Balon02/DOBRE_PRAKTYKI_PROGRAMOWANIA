import keras
from keras import ops, layers

from .block import YoloConv2D, C3k2, SPPF, C2PSA
from .head import YoloDetectionHead

class YoloV11(layers.Layer):
    """Yolov11"""
    def __init__(self, depth:float=0.5, width:float=0.25, max_channels:int=1024, num_classes:int=1, input_res:int=640, add_downsample:bool=False, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.width = width
        self.max_channels = max_channels
        self.input_res = input_res
        self.num_classes = num_classes
        self.add_downsample = add_downsample
        # BACKBONE STACK
        self.backbone_stage_1 = keras.Sequential([
                # -----------THE ONLY DEVIATION TO BETTER HANDLE PICTURES BIGGER THAN ORIGINAL YOLO WAS DESIGNED FOR-------------
                YoloConv2D(filters=self._scale_width(32), kernel_size=3, strides=2) if add_downsample else layers.Identity(), # |
                # ---------------------------------------------------------------------------------------------------------------
                YoloConv2D(filters=self._scale_width(64), kernel_size=3, strides=2),
                YoloConv2D(filters=self._scale_width(128), kernel_size=3, strides=2),
                * [C3k2(dim=self._scale_width(256), use_c3k=False, expansion_ratio=0.25) for _ in range(self._scale_depth(2))],
                YoloConv2D(filters=self._scale_width(256), kernel_size=3, strides=2),
                * [C3k2(dim=self._scale_width(512), use_c3k=False, expansion_ratio=0.25) for _ in range(self._scale_depth(2))],
            ])
        
        self.backbone_stage_2 = keras.Sequential([
                YoloConv2D(filters=self._scale_width(512), kernel_size=3, strides=2),
                * [C3k2(dim=self._scale_width(512), use_c3k=True) for _ in range(self._scale_depth(2))],
            ])
        
        self.backbone_stage_3 = keras.Sequential([
                YoloConv2D(filters=self._scale_width(1024), kernel_size=3, strides=2),
                * [C3k2(dim=self._scale_width(1024), use_c3k=True) for _ in range(self._scale_depth(2))],
                SPPF(dim=self._scale_width(1024), kernel_size=5),
                * [C2PSA(dim=self._scale_width(1024)) for _ in range(self._scale_depth(2))],
            ])
        
        # HEAD STACK
        self.head_upsample_1 = layers.UpSampling2D(size=(2,2), interpolation='nearest')
        self.head_c3k2_1 = keras.Sequential([C3k2(dim=self._scale_width(512), use_c3k=False) for _ in range(self._scale_depth(2))])

        self.head_upsample_2 = layers.UpSampling2D(size=(2,2), interpolation='nearest')
        self.head_c3k2_2 = keras.Sequential([C3k2(dim=self._scale_width(256), use_c3k=False) for _ in range(self._scale_depth(2))])

        self.head_downsample_1 = YoloConv2D(filters=self._scale_width(256), kernel_size=3, strides=2)
        self.head_c3k2_3 = keras.Sequential([C3k2(dim=self._scale_width(512), use_c3k=False) for _ in range(self._scale_depth(2))])

        self.head_downsample_2 = YoloConv2D(filters=self._scale_width(512), kernel_size=3, strides=2)
        self.head_c3k2_4 = keras.Sequential([C3k2(dim=self._scale_width(1024), use_c3k=True) for _ in range(self._scale_depth(2))])

        # DETECT TIP
        strides = ops.array([8,16,32])
        if add_downsample: strides = strides * 2
        self.detect = YoloDetectionHead(num_classes=num_classes, strides=strides, input_res=input_res)

    def _scale_width(self, dim): return min(int(dim*self.width), self.max_channels)

    def _scale_depth(self, repeats): return max(int(repeats*self.depth), 1)

    def call(self, x, training=None):
        if training is None: training = False
        # BACKBONE STAGES
        backbone_stage_1_out = self.backbone_stage_1(x, training=training)
        backbone_stage_2_out = self.backbone_stage_2(backbone_stage_1_out, training=training)
        backbone_stage_3_out = self.backbone_stage_3(backbone_stage_2_out, training=training)
        # HEAD STAGE 1
        upsample_1_out = self.head_upsample_1(backbone_stage_3_out)
        head_c3k2_1_in = ops.concatenate([upsample_1_out, backbone_stage_2_out], axis=-1) # add information from backbone stage 2
        head_c3k2_1_out = self.head_c3k2_1(head_c3k2_1_in, training=training)
        # HEAD STAGE 2 
        upsample_2_out = self.head_upsample_2(head_c3k2_1_out)
        head_c3k2_2_in = ops.concatenate([upsample_2_out, backbone_stage_1_out], axis=-1) # add information from backbone stage 1
        head_c3k2_2_out = self.head_c3k2_2(head_c3k2_2_in, training=training)
        # HEAD STAGE 3
        downsample_1_out = self.head_downsample_1(head_c3k2_2_out, training=training)
        head_c3k2_3_in = ops.concatenate([downsample_1_out, head_c3k2_1_out], axis=-1) # add information from head stage 1
        head_c3k2_3_out = self.head_c3k2_3(head_c3k2_3_in, training=training)
        # HEAD STAGE 4
        downsample_2_out = self.head_downsample_2(head_c3k2_3_out, training=training)
        head_c3k2_4_in = ops.concatenate([downsample_2_out, backbone_stage_3_out], axis=-1) # add information from backbone stage 3
        head_c3k2_4_out = self.head_c3k2_4(head_c3k2_4_in, training=training)
        # DETECTION LAYER
        return self.detect([head_c3k2_2_out, head_c3k2_3_out, head_c3k2_4_out], training=training)
        
    def get_config(self):
        config = super().get_config()
        config.update({"depth": self.depth, "width": self.width, "max_channels": self.max_channels, "input_res": self.input_res, "num_classes": self.num_classes, "add_downsample": self.add_downsample})
        return config

