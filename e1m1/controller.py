"""Converts six Deadly Corridor actions to E1M1 engine controls."""
from corridor.config import ACTIONS

def combat_buttons(index):
    action = ACTIONS[index]
    return {'FORWARD':('MOVE_FORWARD',), 'STRAFE_LEFT':('MOVE_LEFT',),
            'STRAFE_RIGHT':('MOVE_RIGHT',), 'TURN_LEFT':('TURN_LEFT',),
            'TURN_RIGHT':('TURN_RIGHT',), 'SHOOT':('ATTACK',)}[action]
