from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from . import config as C


def gray_resize(screen):
    image = np.asarray(screen)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3:
        image = image[..., :3].astype(np.float32)
        image = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    if image.max(initial=0) > 1.0:
        image /= 255.0
    t = torch.from_numpy(image.copy())[None, None]
    return F.interpolate(t, size=(C.SCREEN_SIZE, C.SCREEN_SIZE), mode='bilinear', align_corners=False)[0, 0].numpy()


class FrameStack:
    def __init__(self):
        self.frames = deque(maxlen=C.FRAME_STACK)

    def reset(self, screen):
        frame = gray_resize(screen)
        self.frames.clear()
        for _ in range(C.FRAME_STACK):
            self.frames.append(frame.copy())
        return self.value()

    def append(self, screen):
        self.frames.append(gray_resize(screen))
        return self.value()

    def value(self):
        return np.stack(self.frames).astype(np.float32)
