import os
import gdown

os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import ops
import jax
import numpy as np
import cv2

from yolov11.yolo import YoloV11
from yolov11.infer_wrapper import YoloV11Inference

from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model
from fast_plate_ocr.core.process import (
    preprocess_image,
    postprocess_output,
    read_and_resize_plate_image,
)


YOLO_WEIGHTS = 'https://drive.google.com/file/d/16RPtgDg3Y_Ql-FAu3bCZ4VnqiNuCkr74/view?usp=drive_link'
EASY_PLATE_OCR_WEIGHTS = 'https://drive.google.com/file/d/19e_7ch1VCeB3iY2M8uNNpxdErz6Ieo1J/view?usp=drive_link'

keras.mixed_precision.set_global_policy("mixed_float16")

class InferencePipeline:
    def __init__(self, batch=1):
        if not os.path.exists('yolo.h5'):
            gdown.download(YOLO_WEIGHTS, 'yolo.h5', quiet=False, fuzzy=True)
        
        if not os.path.exists('ocr.keras'):
            gdown.download(EASY_PLATE_OCR_WEIGHTS, 'ocr.keras', quiet=False, fuzzy=True)

        base = YoloV11(depth=0.5, width=0.25, max_channels=1024, num_classes=1, input_res=1024, add_downsample=False)
        self._yolo = YoloV11Inference(base_model=base, num_classes=1, strides=(8, 16, 32), conf_thres=0.25,)
        self._yolo.load_weights('yolo.h5')
        self._yolo.trainable=False
        self._yolo(ops.zeros((batch, 1024, 1024, 3), dtype='float16'))

        self._plate_cfg = load_plate_config_from_yaml('plate_config.yaml')
        self._ocr = load_keras_model('ocr.keras', self._plate_cfg)
        self._ocr.trainable=False

    @jax.jit
    def _yolo_forward(self, x): return self._yolo(x, training=False)

    @jax.jit
    def _ocr_forward(self, x): return self._ocr(x, training=False)
