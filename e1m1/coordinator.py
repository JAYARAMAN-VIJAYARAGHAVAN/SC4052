"""Hysteretic combat handoff to reduce flickering between policies."""
from . import config as C

class Coordinator:
    def __init__(self): self.mode = 'navigation'
    def decide(self, probability, ammo):
        if self.mode == 'combat':
            if probability < C.ENEMY_RELEASE_THRESHOLD or ammo <= 0:
                self.mode = 'navigation'
        elif probability >= C.ENEMY_THRESHOLD and ammo > 0:
            self.mode = 'combat'
        return self.mode
