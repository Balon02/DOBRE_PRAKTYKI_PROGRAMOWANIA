import os
import gdown

os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import ops
from keras.models import load_model
import jax
import numpy as np
import cv2
from yolov11.yolo import YoloV11
from yolov11.infer_wrapper import YoloV11Inference
from yolov11.block import YoloConv2D, C3k2, SPPF, C2PSA, DFL
from yolov11.head import YoloDetectionHead

from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model
from fast_plate_ocr.core.process import (
    preprocess_image,
    postprocess_output,
    resize_image,
)


YOLO_WEIGHTS = 'https://drive.google.com/file/d/16RPtgDg3Y_Ql-FAu3bCZ4VnqiNuCkr74/view?usp=drive_link'
EASY_PLATE_OCR_WEIGHTS = 'https://drive.google.com/file/d/19e_7ch1VCeB3iY2M8uNNpxdErz6Ieo1J/view?usp=drive_link'

keras.mixed_precision.set_global_policy("mixed_float16")

class InferencePipeline:
    def __init__(self, batch=1):
        if batch < 1:
            raise ValueError("batch must be >= 1")
        self._batch = batch
        # if not os.path.exists('yolo.h5'):
        #     gdown.download(YOLO_WEIGHTS, 'yolo.h5', quiet=False, fuzzy=True, verify=False)
        # 
        # if not os.path.exists('ocr.keras'):
        #     gdown.download(EASY_PLATE_OCR_WEIGHTS, 'ocr.keras', quiet=False, fuzzy=True, verify=False)

        self._yolo = self._load_yolo_model()
        self._yolo_input_res = getattr(self._yolo.base_model, "input_res", 1024)
        self._yolo.trainable=False
        self._yolo(ops.zeros((batch, self._yolo_input_res, self._yolo_input_res, 3), dtype='float16'))

        self._plate_cfg = load_plate_config_from_yaml('plate_config.yaml')
        self._ocr = load_keras_model('ocr.keras', self._plate_cfg)
        self._ocr.trainable=False

    def _load_yolo_model(self):
        model_path = os.environ.get("YOLO_WEIGHTS_PATH", "yolo.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found at {model_path}. Set YOLO_WEIGHTS_PATH.")
        custom_objects = {
            "YoloV11Inference": YoloV11Inference,
            "YoloV11": YoloV11,
            "YoloConv2D": YoloConv2D,
            "C3k2": C3k2,
            "SPPF": SPPF,
            "C2PSA": C2PSA,
            "YoloDetectionHead": YoloDetectionHead,
            "DFL": DFL,
        }
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model

    def _yolo_forward(self, x): return self._yolo(x, training=False)

    def _ocr_forward(self, x): return self._ocr(x, training=False)

    @property
    def batch_size(self) -> int:
        return self._batch

    def _prepare_ocr_image(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if self._plate_cfg.image_color_mode == "grayscale":
            img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return resize_image(
            img,
            img_height=self._plate_cfg.img_height,
            img_width=self._plate_cfg.img_width,
            image_color_mode=self._plate_cfg.image_color_mode,
            keep_aspect_ratio=self._plate_cfg.keep_aspect_ratio,
            interpolation_method=self._plate_cfg.interpolation,
            padding_color=self._plate_cfg.padding_color,
        )

    def _ocr_infer(self, crops: list[np.ndarray]) -> list[str] | None:
        ocr_imgs = []
        for crop in crops:
            img = self._prepare_ocr_image(crop)
            if img is not None:
                ocr_imgs.append(img)
        if not ocr_imgs:
            return None
        batch = preprocess_image(np.stack(ocr_imgs, axis=0))
        y = np.array(self._ocr_forward(batch))
        return postprocess_output(
            model_output=y,
            max_plate_slots=self._plate_cfg.max_plate_slots,
            model_alphabet=self._plate_cfg.alphabet,
            return_confidence=False,
        )

    def infer(self, images):
        # INPUT IMAGE IS ASSUMED TO BE BGR; RAW FROM OPENCV JPEG DECODE
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                imgs = [images]
                single = True
            elif images.ndim == 4:
                imgs = [images[i] for i in range(images.shape[0])]
                single = False
            else:
                return None
        else:
            imgs = list(images)
            single = False

        if not imgs:
            return None

        if len(imgs) > self._batch:
            raise ValueError(f"Received {len(imgs)} images but batch is {self._batch}; caller should chunk requests.")

        yolo_inputs = []
        orig_sizes = []
        for img in imgs:
            if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
                return None
            orig_h, orig_w = img.shape[:2]
            resized = cv2.resize(img, (self._yolo_input_res, self._yolo_input_res))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            yolo_inputs.append(rgb.astype("float16") / 255.0)
            orig_sizes.append((orig_h, orig_w))

        # Pad to fixed batch to keep JAX-compiled shapes stable.
        if len(yolo_inputs) < self._batch:
            pad_needed = self._batch - len(yolo_inputs)
            for _ in range(pad_needed):
                zero_img = np.zeros(
                    (self._yolo_input_res, self._yolo_input_res, 3), dtype="float16"
                )
                yolo_inputs.append(zero_img)
                orig_sizes.append((self._yolo_input_res, self._yolo_input_res))

        yolo_batch = np.stack(yolo_inputs, axis=0)
        preds = np.array(self._yolo_forward(yolo_batch))

        results = [None] * len(imgs)
        crops = []
        crop_indices = []
        for i, pred in enumerate(preds):
            if i >= len(imgs):
                break  # ignore padded items
            confs = pred[:, 4]
            if not np.any(confs > 0):
                continue
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2, _, _ = pred[best_idx]
            orig_h, orig_w = orig_sizes[i]
            x_scale = orig_w / float(self._yolo_input_res)
            y_scale = orig_h / float(self._yolo_input_res)
            x1 = int(max(0, min(orig_w - 1, x1 * x_scale)))
            x2 = int(max(0, min(orig_w, x2 * x_scale)))
            y1 = int(max(0, min(orig_h - 1, y1 * y_scale)))
            y2 = int(max(0, min(orig_h, y2 * y_scale)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = imgs[i][y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_indices.append(i)

        if not crops:
            return None

        plates = self._ocr_infer(crops)
        if not plates:
            return None

        for i, plate in enumerate(plates):
            cleaned = plate.replace("_", "")
            if cleaned:
                results[crop_indices[i]] = cleaned

        if single:
            return results[0]
        return results
