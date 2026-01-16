import math
import os
import warnings
from pathlib import Path
from typing import Callable, Tuple, Optional

import cv2
import numpy as np
import keras
from keras import utils, ops


class YoloDetectionDataset:
    """
    Ultralytics-style dataset loader for YOLO11.

    Exposes train/val PyDataset views yielding (images, targets):
      - images: float32/mixed-policy dtype, scaled to [0,1], shape (B, H, W, 3)
      - targets: float32, shape (B, max_objs, 5) with [cls, x1, y1, x2, y2] (cls will be cast to int in loss)
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 8,
        image_format: str = ".png",
        image_size: Tuple[int, int, int] = (640, 640, 3),
        max_objects: int = 150,
        augment: Optional[Callable] = None,
        debug: bool = False,
        **kwargs
    ):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.train_dir = "train"
        self.val_dir = "val"
        self.debug = debug

        self.batch_size = batch_size
        self.image_format = image_format
        self.image_size = image_size
        self.max_objects = max_objects
        self.augment = augment

        # collect names
        train_label_dir = self.labels_dir / self.train_dir
        val_label_dir = self.labels_dir / self.val_dir
        if not train_label_dir.exists():
            raise FileNotFoundError(f"Train labels directory not found at {train_label_dir}")
        val_names = []
        if not debug:
            if val_label_dir.exists():
                val_names = [n.removesuffix(".txt") for n in os.listdir(val_label_dir) if n.endswith('.txt')]
            else:
                warnings.warn(f"Val labels directory not found at {val_label_dir}, using empty val split.")
        self.name_base = {
            "train": [n.removesuffix(".txt") for n in os.listdir(train_label_dir) if n.endswith('.txt')],
            "val": val_names,
        }

        self._train_view = _DatasetView(self, "train", augment, **kwargs)
        self._val_view = _DatasetView(self, "val", None, **kwargs) if not self.debug else None

    @property
    def train_view(self):
        return self._train_view

    @property
    def val_view(self):
        return self._val_view


class _DatasetView(utils.PyDataset):
    def __init__(self, parent: YoloDetectionDataset, split: str, augment: Optional[Callable], **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.split = split
        self.augment = augment

        self.step_amount = math.ceil(len(parent.name_base[self.split]) / parent.batch_size)
        self.batched_indices = np.reshape(
            np.arange(self.step_amount * parent.batch_size), (self.step_amount, self.parent.batch_size)
        )
        self.batch_validity_mask = np.zeros_like(self.batched_indices)
        self._generate_batch_validity_mask()
        self.epoch = 0

    def _generate_batch_validity_mask(self):
        flat_indices = self.batched_indices.ravel()
        max_valid_index = len(self.parent.name_base[self.split]) - 1
        flat_mask = np.where(flat_indices <= max_valid_index, 1, 0)
        self.batch_validity_mask = np.reshape(flat_mask, self.batched_indices.shape)

    def shuffle_indices(self):
        flat_indices = self.batched_indices.ravel()
        rng = np.random.default_rng(self.epoch)
        flat_indices = rng.permutation(flat_indices)
        self.batched_indices = np.reshape(flat_indices, self.batched_indices.shape)
        self._generate_batch_validity_mask()

    def __len__(self):
        return self.step_amount

    def __getitem__(self, idx):
        current_batch_indices = self.batched_indices[idx]
        current_batch_validity_mask = self.batch_validity_mask[idx]

        imgs = []
        labels_cls = []
        labels_boxes = []

        H, W, _ = self.parent.image_size
        dtype_img = keras.backend.floatx()

        for i in range(len(current_batch_indices)):
            img = np.zeros(self.parent.image_size, dtype="uint8")
            cls_pad = np.zeros((self.parent.max_objects, 1), dtype="int32")
            box_pad = np.zeros((self.parent.max_objects, 4), dtype="float32")

            if current_batch_validity_mask[i]:
                name_base = self.parent.name_base[self.split][current_batch_indices[i]]
                img_path = str(self.parent.images_dir / self.split / (name_base + self.parent.image_format))
                img_read = cv2.imread(img_path)
                if img_read is None:
                    raise FileNotFoundError(f"Image not found at {img_path}")
                img_bgr = cv2.resize(img_read, (W, H))
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # keep RGB throughout

                bboxes = []
                class_labels = []
                label_path = str(self.parent.labels_dir / self.split / (name_base + ".txt"))
                with open(label_path, "r") as file:
                    for box_idx, line in enumerate(file):
                        values = [float(n) for n in line.split()]
                        cls, cx, cy, w, h = values[0], values[1], values[2], values[3], values[4]
                        if box_idx >= self.parent.max_objects:
                            break
                        class_labels.append(int(cls))
                        # convert from normalized cxcywh to xyxy pixels after resize
                        x1 = (cx - w * 0.5) * W
                        y1 = (cy - h * 0.5) * H
                        x2 = (cx + w * 0.5) * W
                        y2 = (cy + h * 0.5) * H
                        bboxes.append([x1, y1, x2, y2])
                        cls_pad[box_idx, 0] = int(cls)
                        box_pad[box_idx] = np.array([x1, y1, x2, y2], dtype="float32")

                bboxes = [[
                        max(0.0, min(W - 1.0, x1)),
                        max(0.0, min(H - 1.0, y1)),
                        max(0.0, min(W - 1.0, x2)),
                        max(0.0, min(H - 1.0, y2)),
                    ] for (x1, y1, x2, y2) in bboxes]

                if self.augment is not None:
                    augmented = self.augment(image=img, bboxes=bboxes, class_labels=class_labels)
                    img = augmented["image"]
                    bboxes = augmented["bboxes"]
                    class_labels = augmented["class_labels"]
                    cls_pad.fill(0)
                    box_pad.fill(0.0)
                    for box_idx, (cls, (x1, y1, x2, y2)) in enumerate(zip(class_labels, bboxes)):
                        if box_idx >= self.parent.max_objects:
                            break
                        cls_pad[box_idx, 0] = int(cls)
                        box_pad[box_idx] = np.array([x1, y1, x2, y2], dtype="float32")
            imgs.append(img)
            labels_cls.append(cls_pad)
            labels_boxes.append(box_pad)

        x_batch = np.array(imgs, dtype="float32") / 255.0
        x_batch = x_batch.astype(dtype_img)
        cls_batch = np.array(labels_cls, dtype="float32")  # stored as float to pack with boxes
        box_batch = np.array(labels_boxes, dtype="float32")
        targets = np.concatenate([cls_batch, box_batch], axis=-1)

        if idx == len(self) - 1:
            self.epoch += 1
            self.shuffle_indices()

        return x_batch, targets
