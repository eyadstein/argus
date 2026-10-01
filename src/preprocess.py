"""
Frame preprocessing for Argus.
"""

import cv2
import numpy as np
from collections import deque


def preprocess_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized


class FrameStacker:
    def __init__(self, k: int = 4, size: int = 84):
        self.k = k
        self.size = size
        self.frames = deque(maxlen=k)

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(first_frame, self.size)
        for _ in range(self.k):
            self.frames.append(processed)
        return self._get_stacked()

    def step(self, new_frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(new_frame, self.size)
        self.frames.append(processed)
        return self._get_stacked()

    def _get_stacked(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)
