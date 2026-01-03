
from __future__ import annotations

from ultralytics import YOLO# type: ignore
import cv2# type: ignore


class Detector(object):

    def __init__(self, weightPath:str="yolov8n.pt") -> None:
        self.model = YOLO(weightPath)
    
    def forward(self, frame):
        return self.model(frame, conf=0.35)[0]




from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # (x_min, y_min, w, h) in pixels
    box: Tuple[int, int, int, int]


class Detectorv2(object):
    def __init__(
        self,
        cfg_path: str,
        weights_path: str,
        names_path: str,
        probability_minimum: float = 0.5,
        nms_threshold: float = 0.3,
        input_size: Tuple[int, int] = (416, 416),
        swap_rb: bool = True,
        use_cuda: bool = False,
    ) -> None:
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.names_path = names_path

        self.probability_minimum = float(probability_minimum)
        self.nms_threshold = float(nms_threshold)
        self.input_size = tuple(input_size)
        self.swap_rb = bool(swap_rb)

        with open(self.names_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f if line.strip()]

        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)

        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names_all = self.net.getLayerNames()
        self.output_layer_names = [layer_names_all[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

        rng = np.random.default_rng(42)
        self.colors = rng.integers(0, 255, size=(len(self.labels), 3), dtype=np.uint8)

    def forward(
        self,
        img: Union[str, np.ndarray],
        draw: bool = True,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
            (output_image, detections)
              - output_image: annotated image if draw=True else original image copy
              - detections: list of Detection objects after NMS
        """
        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise FileNotFoundError(f"Could not read image from path: {img}")
        else:
            image = img
            if not isinstance(image, np.ndarray):
                raise TypeError("img must be a file path (str) or a numpy array (np.ndarray)")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("img ndarray must have shape (H, W, 3)")

        H, W = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=self.input_size,
            swapRB=self.swap_rb,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layer_names)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for out in outputs:
            for det in out:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf < self.probability_minimum:
                    continue
                x_center = int(det[0] * W)
                y_center = int(det[1] * H)
                bw = int(det[2] * W)
                bh = int(det[3] * H)

                x_min = int(x_center - bw / 2)
                y_min = int(y_center - bh / 2)

                boxes.append([x_min, y_min, bw, bh])
                confidences.append(conf)
                class_ids.append(class_id)
        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=self.probability_minimum,
            nms_threshold=self.nms_threshold,
        )

        detections: List[Detection] = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                cid = class_ids[i]
                name = self.labels[cid] if 0 <= cid < len(self.labels) else str(cid)
                detections.append(
                    Detection(
                        class_id=cid,
                        class_name=name,
                        confidence=float(confidences[i]),
                        box=(int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])),
                    )
                )
        out_img = image.copy()
        if draw:
            for det in detections:
                x, y, bw, bh = det.box
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(W - 1, x + bw)
                y2 = min(H - 1, y + bh)

                color = tuple(int(c) for c in self.colors[det.class_id % len(self.colors)])
                cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 3)
                label = f"{det.class_name}: {det.confidence:.3f}"
                cv2.putText(
                    out_img,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return out_img, detections



from typing import List, Tuple, Union
import numpy as np
import cv2
import tensorflow as tf



class Detectorv3(object):
    def __init__(
        self,
        model_dir: str = "models/squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model",
        squirrel_class_id: int = 1,
        min_score: float = 0.5,
    ) -> None:
        self.squirrel_class_id = int(squirrel_class_id)
        self.min_score = float(min_score)

        print("Loading TensorFlow model...")
        saved_model = tf.saved_model.load(model_dir)
        self.model = saved_model.signatures["serving_default"]
        print("Model loaded.")

    def forward(
        self,
        img: Union[str, np.ndarray],
    ) -> Tuple[bool, List[dict]]:
        """
            squirrel_present: bool
            detections: list of dicts with box + score
        """

        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img}")
        else:
            image = img

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        num = int(detections["num_detections"][0])
        classes = detections["detection_classes"][0].numpy()
        scores = detections["detection_scores"][0].numpy()
        boxes = detections["detection_boxes"][0].numpy()

        results = []
        squirrel_found = False

        for i in range(num):
            cls = int(classes[i])
            score = float(scores[i])

            if score < self.min_score:
                continue

            y_min, x_min, y_max, x_max = boxes[i]
            box_px = (
                int(x_min * w),
                int(y_min * h),
                int((x_max - x_min) * w),
                int((y_max - y_min) * h),
            )

            det = {
                "class_id": cls,
                "score": score,
                "box": box_px,
            }
            results.append(det)
            if cls == self.squirrel_class_id:
                squirrel_found = True
        return squirrel_found, results