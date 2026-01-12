import asyncio
import cv2
import torch
torch.backends.cudnn.benchmark = True
from ultralytics import YOLO
from paddleocr import PaddleOCR
from contextlib import nullcontext
import paddle
import numpy as np

DATASET_INFO = '/home/balon/datasets/ORIGINAL_LICENCE_PLATE_DS/annotations.xml'
YOLO_PATH = '/home/balon/source/DOBRE_PRAKTYKI_PROGRAMOWANIA/LP-detection.pt'

class InferencePipeline:
    def __init__(self, detector_input=(1056, 1056, 3), ocr_input=(224, 224, 3), text_det_thresh=0.05, text_det_box_thresh=0.05, text_rec_score_thresh=0.0, flip_color=False, resize_ocr_inputs=False):
        self._gpu_lock = asyncio.Lock()
        self._detector = None
        self._ocr = None

        self._detector_input = detector_input
        self._ocr_input = ocr_input

        self._load_detector()
        self._load_ocr(text_det_thresh, text_det_box_thresh, text_rec_score_thresh)

        self._flip_color = flip_color
        self._resize_ocr_inputs = resize_ocr_inputs

    def _load_detector(self): 
        self._detector = YOLO(YOLO_PATH)
        try: self._detector.model.fuse()
        except: pass
        try: self._detector.model.half()
        except: pass

    def _load_ocr(self, text_det_thresh, text_det_box_thresh, text_rec_score_thresh):
        if paddle.device.is_compiled_with_cuda(): paddle.device.set_device('gpu')
        else: paddle.device.set_device('cpu')
        self._ocr = PaddleOCR(
            lang='en',
            ocr_version='PP-OCRv4',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_rec_score_thresh=text_rec_score_thresh,
        )

    @staticmethod
    def _decode_image_bytes(data: bytes) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

    async def infer(self, image_bytes: bytes):
        async with self._gpu_lock:
            img = self._decode_image_bytes(image_bytes)  # HxWxC, BGR
            img = np.ascontiguousarray(img)

            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
            with ctx:
                results = self._detector.predict(
                    source=img,
                    imgsz=self._detector_input[:2],
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False,
                )

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0: return None

            best = boxes.conf.argmax().item()
            x1, y1, x2, y2 = boxes.xyxy[best].int().tolist()
            crop = img[y1:y2, x1:x2]

            if self._resize_ocr_inputs:
                crop = cv2.resize(crop, self._ocr_input[:2], interpolation=cv2.INTER_LINEAR)

            ocr_result = self._ocr.predict(crop, cls=False)
            return ocr_result
