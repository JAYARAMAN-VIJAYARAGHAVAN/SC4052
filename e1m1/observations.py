"""Separate image preprocessing for policies trained at different resolutions."""
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

class Frames:
    def __init__(self, size, count=4):
        self.size = size
        self.frames = deque(maxlen=count)
        self.count = count

    def update(self, screen, reset=False):
        if screen is None:
            return
        arr = np.asarray(screen)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.moveaxis(arr[:3], 0, -1)
        if arr.ndim == 3:
            arr = arr[..., :3].astype(np.float32).mean(axis=-1)
        else:
            arr = arr.astype(np.float32)
        frame = F.interpolate(torch.from_numpy(arr.copy())[None,None],
                              (self.size,self.size), mode='bilinear',
                              align_corners=False)[0,0].numpy() / 255.0
        if reset or not self.frames:
            self.frames.clear()
            for _ in range(self.count): self.frames.append(frame.copy())
        else:
            self.frames.append(frame)

    def tensor(self):
        return torch.from_numpy(np.stack(self.frames).astype(np.float32))[None]
