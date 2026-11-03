
"""Classical steering with basic obstacle recovery; never chooses a room."""
import math
import vizdoom as vzd
from . import config as C

BUTTONS = [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD,
           vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT]
ACTION_NAMES = ['TURN_LEFT', 'TURN_RIGHT', 'MOVE_FORWARD', 'MOVE_LEFT', 'MOVE_RIGHT']


class SteeringController:
    def __init__(self, game):
        self.game = game

    def pose(self):
        g = self.game.get_game_variable
        return (float(g(vzd.GameVariable.POSITION_X)),
                float(g(vzd.GameVariable.POSITION_Y)),
                math.radians(float(g(vzd.GameVariable.ANGLE)) % 360))

    def navigate(self, target, observe, max_actions, stop_when):
        """Returns (outcome, native_reward, trace).

        Outcomes: arrived, game_finished, stuck, waypoint_reached, timeout.
        A waypoint alone is NOT proof of arrival in the selected sector.
        """
        total_reward = 0.
        trace = []
        blocked = 0
        recovery_steps = 0
        recovery_side = 3
        for i in range(max_actions):
            if self.game.is_episode_finished():
                return 'game_finished', total_reward, trace
            x, y, heading = self.pose()
            desired = math.atan2(target[1] - y, target[0] - x)
            error = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
            if recovery_steps:
                action = recovery_side
                recovery_steps -= 1
            elif abs(error) > .18:
                action = 0 if error > 0 else 1
            else:
                action = 2
            controls = [0] * len(BUTTONS)
            controls[action] = 1
            native = float(self.game.make_action(controls, C.ACTION_TICS))
            total_reward += native
            x2, y2, _ = self.pose()
            moved = math.dist((x2, y2), (x, y))
            if action in (2, 3, 4) and moved < .5:
                blocked += 1
            elif moved >= .5:
                blocked = 0
            info = observe(action, native)
            trace.append(info)
            if stop_when(info):
                return 'arrived', total_reward, trace
            if self.game.is_episode_finished():
                return 'game_finished', total_reward, trace
            if blocked >= 3:
                # Try lateral movement, alternating sides. This is a limited
                # recovery heuristic, not a collision-aware path planner.
                if recovery_steps == 0:
                    recovery_side = 3 if (i // 3) % 2 == 0 else 4
                    recovery_steps = 3
                if blocked >= 12:
                    return 'stuck', total_reward, trace
            if math.dist((x2, y2), target) < 8 and i > 3:
                # The caller can retry with a different waypoint later.
                return 'waypoint_reached', total_reward, trace
        return 'timeout', total_reward, trace
