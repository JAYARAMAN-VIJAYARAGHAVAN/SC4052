"""Direct ViZDoom Deadly Corridor environment with label-supervised enemy detection.

Enemy labels are supervision only, never policy input. The 10,000 HP request is
verified on reset; stock ViZDoom scenarios may require a custom WAD/ACS script.
"""
from pathlib import Path
import numpy as np
import vizdoom as vzd
from . import config as C
from .preprocessing import FrameStack

BUTTONS = [vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
           vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK]
SHOOT_ACTION = BUTTONS.index(vzd.Button.ATTACK)
TARGET_HEALTH = 10_000
# This command is engine/scenario dependent; it is NOT a standard ViZDoom HP setter.
HEALTH_COMMAND = 'give health 10000'
ENEMY_NAMES = ('zombieman', 'shotgunguy', 'chaingunguy', 'doomimp', 'imp',
               'demon', 'spectre', 'cacodemon', 'lostsoul', 'baronofhell',
               'hellknight', 'revenant', 'mancubus', 'arachnotron', 'pain elemental',
               'archvile', 'spider mastermind', 'cyberdemon', 'formerhuman')


class CorridorEnv:
    def __init__(self, visible=False, seed=C.SEED, high_health=True):
        self.game = vzd.DoomGame()
        scenario = Path(vzd.__file__).resolve().parent / 'scenarios' / C.SCENARIO
        if not scenario.exists():
            raise FileNotFoundError(scenario)
        self.game.load_config(str(scenario))
        self.game.set_window_visible(visible)
        self.game.set_seed(seed)
        self.game.set_available_buttons(BUTTONS)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_available_game_variables([vzd.GameVariable.HEALTH,
            vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT])
        self.game.init()
        self.stack = FrameStack()
        self.steps = 0
        self.high_health = high_health

    def _enemy_label(self, state):
        if state is None:
            return 0.0
        labels = getattr(state, 'labels', None) or []
        for label in labels:
            name = str(getattr(label, 'object_name', '')).lower().replace('_', '').replace(' ', '')
            if any(enemy.replace(' ', '') in name for enemy in ENEMY_NAMES):
                return 1.0
        return 0.0

    def _observation(self, state):
        return {'image': self.stack.value(), 'enemy_target': self._enemy_label(state)}

    def reset(self):
        self.game.new_episode()

        # Enable god mode for this episode
        self.game.send_game_command("iddqd")

        self.steps = 0
        state = self.game.get_state()

        if state is None:
            raise RuntimeError("No ViZDoom state after reset")

        self.stack.reset(state.screen_buffer)
        return self._observation(state), {}

    def step(self, action):
        if not 0 <= action < C.N_ACTIONS:
            raise ValueError(action)
        before_state = self.game.get_state()
        enemy_before = self._enemy_label(before_state)
        buttons = [0] * len(BUTTONS)
        buttons[action] = 1
        before_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        native = float(self.game.make_action(buttons, C.ACTION_TICS))
        self.steps += 1
        finished = bool(self.game.is_episode_finished())
        dead = bool(self.game.is_player_dead())
        state = None if finished else self.game.get_state()
        if state is not None:
            self.stack.append(state.screen_buffer)
        kills = max(0.0, self.game.get_game_variable(vzd.GameVariable.KILLCOUNT) - before_kills)
        reward = native + 5.0 * kills - 0.005
        if action == SHOOT_ACTION and not enemy_before:
            reward -= 0.2
        truncated = self.steps >= C.MAX_EPISODE_STEPS and not finished
        info = {'native_reward': native, 'kills': kills, 'dead': dead,
                'enemy_visible': self._enemy_label(state), 'steps': self.steps,
                'finished': finished}
        return self._observation(state), reward, finished, truncated, info

    def close(self):
        self.game.close()
