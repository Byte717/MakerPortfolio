from typing import * 
import numpy as np
import cv2
from dataclasses import dataclass
from cv2.typing import MatLike

@dataclass
class LabeledImg:
    img: MatLike
    topLeftPt:Tuple[int,int]
    deltaX:int
    deltaY:int

def crop(mat, topLeft:Tuple[int,int], w:int, h:int):
    return mat[topLeft[1]:topLeft[1]+h , topLeft[0]:topLeft[0]+w]

def tuneValues(imgs:List["LabeledImg"]):
    rng = np.random.default_rng(0)
    per_img_cap = 5000

    chunks = []
    for li in imgs:
        roi = crop(li.img, li.topLeftPt, li.deltaX, li.deltaY)
        if roi.size == 0:
            continue

        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        if roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        p = roi.reshape(-1, 3)
        if len(p) > per_img_cap:
            idx = rng.choice(len(p), size=per_img_cap, replace=False)
            p = p[idx]
        chunks.append(p)

    if not chunks:
        raise ValueError("No valid ROIs found")

    X = np.concatenate(chunks, axis=0).astype(np.int16)

    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    lower = np.clip(np.floor(q1), 0, 255).astype(np.uint8)
    upper = np.clip(np.ceil(q3), 0, 255).astype(np.uint8)

    return lower, upper