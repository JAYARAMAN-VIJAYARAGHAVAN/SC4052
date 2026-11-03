from collections import Counter
import math
from . import config as C

class ExplorationMemory:
    def __init__(self):self.cells=Counter();self.rooms=Counter();self.transitions=Counter()
    def observe(self,x,y,room):
        cell=(math.floor(x/C.GRID_SIZE),math.floor(y/C.GRID_SIZE))
        new_cell=self.cells[cell]==0
        self.cells[cell]+=1
        new_room=room is not None and self.rooms[room]==0
        if room is not None:self.rooms[room]+=1
        return new_cell,new_room
    def room_count(self,room):return self.rooms[room] if room is not None else 0
