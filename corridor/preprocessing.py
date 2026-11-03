from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from . import config as C

def gray84(screen):
    arr = np.asarray(screen)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr[:3], 0, -1)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=-1)
    else:
        arr = arr.astype(np.float32)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return F.interpolate(x, (C.SCREEN_SIZE, C.SCREEN_SIZE), mode='bilinear', align_corners=False)[0, 0].numpy() / 255.0

class FrameStack:
    def __init__(self): self.frames = deque(maxlen=C.FRAME_STACK)
    def reset(self, frame):
        f = gray84(frame)
        self.frames.clear()
        for _ in range(C.FRAME_STACK): self.frames.append(f.copy())
    def append(self, frame): self.frames.append(gray84(frame))
    def value(self): return np.stack(self.frames).astype(np.float32)
